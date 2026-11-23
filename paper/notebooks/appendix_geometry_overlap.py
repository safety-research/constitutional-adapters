"""appendix_geometry_overlap: the overlap figure of the control-directions appendix (Figure 10, figA_cK_overlap), read from the committed
sidecar `data/appendix_geometry_overlap.json` (written by `data/build_appendix_geometry_overlap.py`).

    import appendix_geometry_overlap as GO
    go = GO.load_geometry_overlap()          # refuses, naming every problem, unless the file is whole and self-consistent
    GO.overlap_figure(go)                    # figA_cK_overlap
    GO.table_overlap(go); GO.text_numbers(go)
    GO.registry_check(go, objects)           # every object id against the registry's object table (appendix_geometry_forward's rule)

THE OBJECTS are the paper's Qwen3.5-4B base-to-instruct fits, seeds 1-3: theta_C (constitution) and theta_K (control), as a LoRA and as
a ReLU steer, with a safety-refusal LoRA and ReLU steer (distilled from a safety-refusal system prompt on the instruct checkpoint) as an
unrelated reference. Every object is pinned by sha256, the registry's object_id.

figA_cK_overlap (weights only, no judge):
  (a) LoRA, per LoRA layer: the share of one object's update ||dW_l||_F^2 inside another object's rank-4 output space (span of B_l, where
      it writes; left) or input space (span of A_l, where it reads; right), log scale, in percent. A marker is the mean over a series'
      pairs and its bar their range: theta_C in theta_K and theta_K in theta_C (9 seed pairs), a seed in another seed of the same object
      (6 ordered pairs, the seed ceiling), theta_C or theta_K in the safety LoRA (3). Filled markers and solid lines are the share of
      theta_C's update, open markers and dashed lines theta_K's. Grey: per layer, the 95% range when the other subspace is a random
      4-dimensional one (the lowest 2.5th to the highest 97.5th percentile over the six theta_C and theta_K seeds).
  (b) ReLU steers, per layer: the cosine between two objects' read (gate) vectors (left) and write vectors (right); a line is the mean
      over a series' pairs and its shading their range: theta_C against theta_K (9), seed pairs of one object (3), the safety steer
      against each object's 3 seeds. Grey: +-2/sqrt(d), two standard deviations of the cosine between random vectors.

THE STYLE is the manuscript's (`appendix_geometry_style`): its 5.5 x 4.44 in canvas with every axis, legend row and panel letter placed in
inches, its fonts and palette (theta_C green, theta_K blue, the safety reference yellow, a pair of the two neutral ink), saved at the
canvas as pdf, a 300 dpi png and svg by `GS.save`, so the png it writes is the manuscript's pixel for pixel.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import appendix_geometry_forward as GF
import appendix_geometry_style as GS

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.join(HERE, "data", "appendix_geometry_overlap.json")
COSINES = os.path.join(HERE, "data", "relu_arditi_cosines.json")
SCHEMA = "appendix_geometry_overlap/1"
BUILDER = "paper/notebooks/data/build_appendix_geometry_overlap.py"
FIGURE = "figA_cK_overlap"
SEEDS = ("1", "2", "3")
LORA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
N_LAYERS = 32
LORA_SERIES = {"CinK": 9, "KinC": 9, "CC": 6, "KK": 6, "CinS": 3, "KinS": 3, "SinC": 3, "SinK": 3}
RELU_SERIES = {"CK": 9, "CC": 3, "KK": 3, "SC": 3, "SK": 3}
LAYER12_REL = 1e-7                            # the sidecar's cosines carry 8 significant digits; the committed cosine data full float32
NOT_DRAWN = ("SinC", "SinK")                  # the safety LoRA's update in theta_C's or theta_K's span: in the tables, not the figure
ROLE_LABEL = {("relu", "C"): "constitution_relu", ("relu", "K"): "control_relu", ("relu", "S"): "safety_relu",
              ("lora", "C"): "constitution_lora", ("lora", "K"): "control_lora", ("lora", "S"): "safety_lora"}
FIT_MODEL = {"C": "qwen35_4b_base", "K": "qwen35_4b_base", "S": "qwen35_4b_it"}
DASH = (0, (3.0, 1.6))
# panel (a): series -> (legend label, colour key, filled, x offset in layer units); filled and solid = the share of theta_C's update,
# open and dashed = theta_K's; colour key: "ink" for a pair of theta_C and theta_K, else the object whose seeds are paired ("C", "K") or
# the safety reference ("S"), in appendix_geometry_style's palette
SERIES_A = {"CinK": (r"$\theta_C$ in $\theta_K$ (9 seed pairs)", "ink", True, -0.55),
            "KinC": (r"$\theta_K$ in $\theta_C$ (9 seed pairs)", "ink", False, 0.55),
            "CC": (r"$\theta_C$ in another $\theta_C$ seed (6)", "C", True, -0.55),
            "KK": (r"$\theta_K$ in another $\theta_K$ seed (6)", "K", False, 0.55),
            "CinS": (r"$\theta_C$ in the safety LoRA (3)", "S", True, -0.55),
            "KinS": (r"$\theta_K$ in the safety LoRA (3)", "S", False, 0.55)}
# panel (b): series -> (legend label, colour key, dashed, shading alpha)
SERIES_B = {"CK": (r"$\theta_C$ vs $\theta_K$ (9 seed pairs)", "ink", False, 0.16),
            "CC": (r"$\theta_C$ vs $\theta_C$ (3 seed pairs)", "C", False, 0.16),
            "KK": (r"$\theta_K$ vs $\theta_K$ (3 seed pairs)", "K", True, 0.16),
            "SC": (r"safety vs $\theta_C$ (3 seeds)", "S", False, 0.28),
            "SK": (r"safety vs $\theta_K$ (3 seeds)", "S", True, 0.28)}
# the label over each sub-axis (the manuscript's wording, the read vectors under the objects' own field name)
SPACE_TITLE = {"write": "Output (write) space, span of B", "read": "Input (read) space, span of A"}
VECTOR_TITLE = {"read": "Read vectors (gate_w)", "write": "Write vectors (directions)"}
COLUMNS = {"lora_overlap": ["space", "layer", "series", "measured", "span_of", "pair_type", "share"],
           "lora_null": ["measured", "layer", "space", "mean", "q0.025", "q0.5", "q0.975", "q0.99"],
           "relu_cosine": ["vector", "layer", "series", "a", "b", "cosine"]}


class GeometryOverlapSidecarError(ValueError):
    pass


# ---- reading and checking the sidecar -----------------------------------------------------------------------------------------------
def frame(sc: dict, name: str) -> pd.DataFrame:
    b = sc[name]
    return pd.DataFrame(b["rows"], columns=b["columns"])


def _key(role: str, family: str, seed: str) -> str:
    return f"{role}_{family}:{('s' + seed) if seed else '-'}"


def layer12_problems(sc: dict, cosines_path: str = COSINES) -> list:
    """Panel (b) at layer 12 against the committed layer-12 cosine data (`relu_arditi_cosines.json`), which the same objects' read and
    write vectors gave: every cross pair and seed pair equal to the sidecar's 8 significant digits, and each of its ReLU objects the
    sidecar's object of that role and seed."""
    with open(cosines_path, encoding="utf-8") as fh:
        cos = json.load(fh)
    lab = {"constitution_relu": "C", "control_relu": "K", "safety_relu": "S"}
    ids = {o["key"]: o["object_id"] for o in sc["objects"]}
    p = []
    for e in cos["entries"]:
        r = lab.get(e["object_label"])
        if r and (r == "S" or e["fit_model_id"] == "qwen35_4b_base"):
            k = _key(r, "relu", "" if r == "S" else e["seed"].lstrip("s"))
            if ids.get(k) != e["object_id"]:
                p.append(f"{k}: the sidecar pins {str(ids.get(k))[:12]}, the cosine data {e['object_id'][:12]}")
    rc = frame(sc, "relu_cosine")
    rc = rc[rc.layer.eq(12)].set_index(["vector", "a", "b"]).cosine
    for e in cos["cross_pairs"] + cos["pairs"]:
        oa, ob = e.get("object_a", e.get("object_label")), e.get("object_b", e.get("object_label"))
        if oa in lab and ob in lab:
            a, b = _key(lab[oa], "relu", e["seed_a"].lstrip("s")), _key(lab[ob], "relu", e["seed_b"].lstrip("s"))
            for vec in ("read", "write"):
                v = rc.get((vec, a, b))
                if v is None or not math.isclose(v, e[vec], rel_tol=LAYER12_REL, abs_tol=1e-9):
                    p.append(f"layer-12 {vec} cosine {a} vs {b}: {rc.get((vec, a, b))!r}, the cosine data {e[vec]!r}")
    return p


def geometry_overlap_problems(sc: dict) -> list:
    """Every reason the sidecar cannot be drawn from, as sentences; empty when it is whole. Checks the stamp, the provenance, the
    object pins and their registry claims, each table's shape, coverage and range, the null's expected shares, and panel (b) at layer
    12 against the committed cosine data."""
    p = []
    if sc.get("schema") != SCHEMA:
        return [f"schema {sc.get('schema')!r}, expected {SCHEMA!r}"]
    if (sc.get("judge") or {}).get("model") != "none":
        p.append(f"judge stamp {sc.get('judge')!r}: this figure is judge-free and the stamp must say model 'none'")
    pv = sc.get("provenance") or {}
    for k in ("generated_utc", "script", "script_sha256", "source_files", "registry", "checks"):
        if k not in pv:
            p.append(f"provenance lacks {k!r}")
    if pv.get("script") != BUILDER:
        p.append(f"provenance.script {pv.get('script')!r}, expected {BUILDER!r}")
    files = pv.get("source_files") or []
    if not files or any(not GF.HEX.match(str(f.get("sha256", ""))) or not f.get("name") for f in files):
        p.append("provenance.source_files must name every source with its sha256")
    leaks = sorted({s for s in GF._strings(sc) if s.startswith(("/", "~")) or "/home/" in s})
    if leaks:
        p.append(f"absolute or home paths in the sidecar: {leaks[:3]}")
    objs = sc.get("objects") or []
    ids = [o.get("object_id", "") for o in objs]
    if len(set(ids)) != len(ids) or not all(GF.HEX.match(i) for i in ids):
        p.append("objects: every object_id must be a distinct sha256")
    have = sorted((o.get("family"), o.get("role"), o.get("seed")) for o in objs)
    want = sorted({(f, r, s) for f in ("relu", "lora") for r in ("C", "K") for s in SEEDS} | {("relu", "S", ""), ("lora", "S", "")})
    if have != want:
        p.append(f"objects: {have}, expected {want}")
    for o in objs:
        reg = o.get("registry") or {}
        if o.get("key") != _key(o.get("role"), o.get("family"), o.get("seed")):
            p.append(f"object {o.get('object_id', '')[:12]}: key {o.get('key')!r}")
        lab, fit = ROLE_LABEL.get((o.get("family"), o.get("role"))), FIT_MODEL.get(o.get("role"))
        if reg.get("paper_label") != lab or reg.get("fit_model_id") != fit:
            p.append(f"object {o.get('object_id', '')[:12]} ({o.get('key')}): registry claim {reg.get('paper_label')}/{reg.get('fit_model_id')}, "
                     f"expected {lab}/{fit}")
    for name, cols in COLUMNS.items():
        b = sc.get(name)
        if not isinstance(b, dict) or b.get("columns") != cols or not b.get("rows") or any(len(r) != len(cols) for r in b["rows"]):
            p.append(f"table {name}: columns {None if not isinstance(b, dict) else b.get('columns')}, expected {cols}, with rows of that width")
    if p:
        return p
    keys = {o["key"] for o in objs}
    lo = frame(sc, "lora_overlap")
    n = lo.groupby(["space", "layer", "series"]).size().to_dict()
    if n != {(sp, l, se): k for sp in ("write", "read") for l in LORA_LAYERS for se, k in LORA_SERIES.items()}:
        p.append("lora_overlap must hold every space x LoRA layer x series with its number of pairs (9, 9, 6, 6, 3, 3, 3, 3)")
    if not set(lo.measured) | set(lo.span_of) <= {k for k in keys if "_lora:" in k}:
        p.append("lora_overlap names an object that is not a pinned LoRA")
    if not lo.share.between(0, 1).all():
        p.append("lora_overlap: a share outside [0, 1]")
    rc = frame(sc, "relu_cosine")
    n = rc.groupby(["vector", "layer", "series"]).size().to_dict()
    if n != {(v, l, se): k for v in ("read", "write") for l in range(N_LAYERS) for se, k in RELU_SERIES.items()}:
        p.append("relu_cosine must hold every vector x layer x series with its number of pairs (9, 3, 3, 3, 3)")
    if not set(rc.a) | set(rc.b) <= {k for k in keys if "_relu:" in k}:
        p.append("relu_cosine names an object that is not a pinned ReLU steer")
    if not rc.cosine.between(-1, 1).all():
        p.append("relu_cosine: a cosine outside [-1, 1]")
    nl = frame(sc, "lora_null")
    lora_keys = sorted(k for k in keys if "_lora:" in k)
    if (sorted(nl[["measured", "layer", "space"]].apply(tuple, axis=1))
            != sorted((k, l, sp) for k in lora_keys for l in LORA_LAYERS for sp in ("write", "read"))):
        p.append("lora_null must hold every LoRA x LoRA layer x space once")
    exp = sc["lora_null"].get("expected") or {}
    rank = sc.get("rank")
    if not (rank and math.isclose(exp.get("write", 0), rank / sc.get("d_model", 1)) and math.isclose(exp.get("read", 0), rank / sc.get("d_in", 1))):
        p.append(f"lora_null.expected {exp}: a random rank-{rank} subspace's share is rank / d_model (write) and rank / d_in (read)")
    if ((nl["q0.025"] > nl["q0.5"]) | (nl["q0.5"] > nl["q0.975"]) | (nl["q0.975"] > nl["q0.99"])).any():
        p.append("lora_null: quantiles out of order")
    if p:
        return p
    return layer12_problems(sc)


def load_geometry_overlap(path: str = SIDECAR) -> dict:
    """The sidecar at `path`, after `geometry_overlap_problems`; raises GeometryOverlapSidecarError naming every problem."""
    with open(path, encoding="utf-8") as fh:
        sc = json.load(fh)
    problems = geometry_overlap_problems(sc)
    if problems:
        raise GeometryOverlapSidecarError(f"{os.path.basename(path)}: " + "; ".join(problems))
    return sc


def registry_check(sc: dict, objects: pd.DataFrame) -> pd.DataFrame:
    """One row per pinned object with `known`: the registry's object table (registry_nb.OBJECTS_SQL) holds that id with the paper label,
    fit checkpoint, seed and derivation the sidecar claims (appendix_geometry_forward.registry_check, the same rule)."""
    return GF.registry_check(sc, objects)


# ---- the panels' numbers ------------------------------------------------------------------------------------------------------------
def summary_frame(sc: dict) -> pd.DataFrame:
    """Per panel ('a' LoRA, 'b' ReLU), space ('write'/'read' for LoRA spaces, 'read'/'write' for ReLU vectors), series and layer: the
    number of pairs, their mean (the marker or line) and their min and max (the bar or shading). Shares as fractions, not percent."""
    lo = frame(sc, "lora_overlap").rename(columns={"share": "value"}).assign(panel="a")
    rc = frame(sc, "relu_cosine").rename(columns={"vector": "space", "cosine": "value"}).assign(panel="b")
    both = pd.concat([lo[["panel", "space", "series", "layer", "value"]], rc[["panel", "space", "series", "layer", "value"]]])
    g = both.groupby(["panel", "space", "series", "layer"]).value
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "min": g.min(), "max": g.max()}).reset_index()


def band_frame(sc: dict) -> pd.DataFrame:
    """Panel (a)'s grey band per space and LoRA layer: the lowest 2.5th and highest 97.5th percentile of the random-subspace null over
    the six theta_C and theta_K seeds, and the mean of their null means."""
    nl = frame(sc, "lora_null")
    nl = nl[nl.measured.str[0].isin(["C", "K"])]
    g = nl.groupby(["space", "layer"])
    return pd.DataFrame({"lo": g["q0.025"].min(), "hi": g["q0.975"].max(), "null_mean": g["mean"].mean()}).reset_index()


def random_cosine_band(sc: dict) -> float:
    """Panel (b)'s grey band: two standard deviations of the cosine between random vectors in the residual width, 2 / sqrt(d)."""
    return 2.0 / math.sqrt(sc["d_model"])


def table_overlap(sc: dict) -> pd.DataFrame:
    """Per panel, space and series: the pairs per layer, the per-layer mean's min and max over layers (with the layers), its mean over
    layers, and the single values' min and max; shares in percent, cosines as cosines. The safety LoRA's update in theta_C's or
    theta_K's span (SinC, SinK) is listed, not drawn."""
    s = summary_frame(sc)
    rows = []
    for panel, space, series_all in (("a", "write", LORA_SERIES), ("a", "read", LORA_SERIES), ("b", "read", RELU_SERIES),
                                     ("b", "write", RELU_SERIES)):
        k = 100.0 if panel == "a" else 1.0
        for series in series_all:
            g = s[s.panel.eq(panel) & s.space.eq(space) & s.series.eq(series)].set_index("layer")
            rows.append(dict(panel=panel, space=space, series=series, drawn=series not in NOT_DRAWN, n=int(g.n.iat[0]),
                             layer_mean_min=k * g["mean"].min(), at_min=int(g["mean"].idxmin()), layer_mean_max=k * g["mean"].max(),
                             at_max=int(g["mean"].idxmax()), mean_over_layers=k * g["mean"].mean(), value_min=k * g["min"].min(),
                             value_max=k * g["max"].max()))
    return pd.DataFrame(rows)


def text_numbers(sc: dict) -> pd.DataFrame:
    """The numbers the appendix text quotes from this figure's data, recomputed from the sidecar: key, value and how."""
    s, b = summary_frame(sc), band_frame(sc)
    out = []

    def put(k, v, how):
        out.append(dict(key=k, value=v, how=how))

    def lm(panel, space, series):
        return s[s.panel.eq(panel) & s.space.eq(space) & s.series.eq(series)]["mean"]

    for se in ("CinK", "KinC"):
        put(f"lora_write_{se}_mean_over_layers_pct", round(100 * lm("a", "write", se).mean(), 2),
            f"share of {'theta_C' if se == 'CinK' else 'theta_K'}'s update in the other's output space, pair mean, mean over layers")
    both = pd.concat([lm("a", "write", "CinK"), lm("a", "write", "KinC")])
    put("lora_write_CK_per_layer_pct", (round(100 * both.min(), 2), round(100 * both.max(), 2)),
        "the same two shares, per-layer pair means, lowest and highest")
    put("lora_write_random_null_mean_pct", round(100 * b[b.space.eq("write")].null_mean.mean(), 3),
        "a random 4-dimensional output space: the null's mean share")
    saf = pd.concat([lm("a", "write", "CinS"), lm("a", "write", "KinS")])
    put("lora_write_in_safety_per_layer_pct", (round(100 * saf.min(), 2), round(100 * saf.max(), 2)),
        "theta_C's or theta_K's update in the unrelated safety LoRA's output space, per-layer means")
    cei = pd.concat([lm("a", "write", "CC"), lm("a", "write", "KK")])
    put("lora_write_seed_ceiling_per_layer_pct", (round(100 * cei.min(), 1), round(100 * cei.max(), 1)),
        "a seed's update in another seed's output space of the same object, per-layer means")
    put("relu_write_CK_max", round(lm("b", "write", "CK").max(), 3), "cosine of theta_C's and theta_K's ReLU write vectors, largest per-layer pair mean")
    put("relu_read_CK_max", round(lm("b", "read", "CK").max(), 3), "same for the read vectors")
    put("relu_write_safety_max", round(pd.concat([lm("b", "write", "SC"), lm("b", "write", "SK")]).max(), 3),
        "cosine of the safety steer's write vectors with either object's, largest per-layer mean")
    return pd.DataFrame(out)


def provenance_line(sc: dict) -> str:
    pv = sc["provenance"]
    return (f"{os.path.basename(SIDECAR)}: {len(sc['lora_overlap']['rows'])} LoRA overlaps, {len(sc['relu_cosine']['rows'])} ReLU cosines and "
            f"{len(sc['lora_null']['rows'])} random-subspace null rows ({sc['lora_null']['n_draws']:,} draws per object, seed "
            f"{sc['lora_null']['seed']}); {len(sc['objects'])} objects pinned by sha256 and bound to the registry (head "
            f"{pv['registry']['head_at_build']}); layer 12 equals {os.path.basename(COSINES)}; {len(pv['source_files'])} source files; "
            f"judge: none (judge-free); built {pv['generated_utc']} by {pv['script']}")


# ---- figA_cK_overlap ----------------------------------------------------------------------------------------------------------------
# The manuscript's canvas (appendix_geometry_style): 5.5 x 4.44 in; per panel a legend row whose top edge is the panel's top (0.00 in for
# (a), 2.30 in for (b), left edge 0.50 in) and two sub-axes 2.20 in wide, 0.62 in from the left and 0.32 in apart, their top edge 0.55 in
# below the legend's (1.38 in tall in (a), 1.30 in in (b)); the panel letter at 0.04 in from the left, level with the legend's top.
CANVAS_IN = (GS.TEXTWIDTH_IN, 4.44)
PANEL_TOP_IN = {"a": 0.0, "b": 2.30}
PANEL_AXES_IN = {"a": 1.38, "b": 1.30}
AXES_DROP_IN, LEFT_IN, AXW_IN, GAP_IN, LEGEND_X_IN, LETTER_X_IN = 0.55, 0.62, 2.20, 0.32, 0.50, 0.04
SERIES_COLOR = {"ink": GS.INK, "C": GS.GREEN, "K": GS.BLUE, "S": GS.YELLOW}
BAND_LABEL_A = "random 4-dim subspaces"
BAND_LABEL_B = r"$\pm 2/\sqrt{d}$"


def _colour(k: str) -> str:
    """A series' colour: neutral ink for a pair of theta_C and theta_K, else the object's (appendix_geometry_style's palette)."""
    return SERIES_COLOR[k]


def _band_steps(layers, lo, hi, x0, x1):
    """The band's outline, carried flat from the first and last layer to the axis ends."""
    return np.r_[x0, layers, x1], np.r_[lo[0], lo, lo[-1]], np.r_[hi[0], hi, hi[-1]]


def _two_axes(fig, fx, fy, top, h):
    H = CANVAS_IN[1]
    axes = [fig.add_axes([fx(LEFT_IN), fy(top + h), fx(AXW_IN), h / H])]
    axes.append(fig.add_axes([fx(LEFT_IN + AXW_IN + GAP_IN), fy(top + h), fx(AXW_IN), h / H], sharey=axes[0]))
    return axes


def _panel_a(fig, fx, fy, s, band):
    top = PANEL_TOP_IN["a"]
    axes = _two_axes(fig, fx, fy, top + AXES_DROP_IN, PANEL_AXES_IN["a"])
    layers = np.array(LORA_LAYERS)
    for ax, space in zip(axes, ("write", "read")):
        b = band[band.space.eq(space)].sort_values("layer")
        xs, lo, hi = _band_steps(layers, 100 * b.lo.values, 100 * b.hi.values, 1, 33)
        ax.fill_between(xs, lo, hi, color=GS.BAND, lw=0, zorder=0)
        ax.text(32.7, 100 * b.lo.min() * 0.88, BAND_LABEL_A, ha="right", va="top", fontsize=GS.FS_SMALL - 0.8, color=GS.MUTED, zorder=1)
        for series, (_lab, ck, filled, off) in SERIES_A.items():
            col, ls = _colour(ck), "-" if filled else DASH
            q = s[s.panel.eq("a") & s.space.eq(space) & s.series.eq(series)].sort_values("layer")
            x = q.layer.values + off
            ax.vlines(x, 100 * q["min"].values, 100 * q["max"].values, color=col, lw=0.8, zorder=2)
            ax.plot(x, 100 * q["mean"].values, color=col, lw=GS.LW_THIN, ls=ls, zorder=3)
            ax.plot(x, 100 * q["mean"].values, ls="none", marker="s", ms=GS.MS * 0.85, zorder=4,
                    mfc=col if filled else GS.SURFACE, mec=GS.SURFACE if filled else col, mew=0.5 if filled else 0.9)
        ax.set_yscale("log")
        ax.set_xticks(layers)
        ax.set_xlim(1, 33)
        ax.set_xlabel("LoRA layer", labelpad=1)
        GS.grid(ax, "y")
        GS.facet(ax, SPACE_TITLE[space], y=1.08)
    axes[0].set_ylim(0.008, 100)
    axes[0].set_yticks([0.01, 0.1, 1, 10, 100])
    axes[0].set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    axes[0].yaxis.set_minor_locator(plt.NullLocator())
    axes[0].set_ylabel("Share of " + r"$\|\Delta W\|^2$" + " inside\nthe other's span (%)", labelpad=2)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    hs = []
    for lab, ck, filled, _off in SERIES_A.values():
        col = _colour(ck)
        hs.append(Line2D([], [], color=col, lw=GS.LW_THIN, ls="-" if filled else DASH, marker="s", ms=GS.MS * 0.85,
                         mfc=col if filled else GS.SURFACE, mec=GS.SURFACE if filled else col, mew=0.5 if filled else 0.9, label=lab))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(fx(LEGEND_X_IN), fy(top)), ncol=3, frameon=False, columnspacing=1.0,
               handletextpad=0.35, handlelength=2.0, fontsize=GS.FS_SMALL)
    return axes


def _panel_b(fig, fx, fy, s, rb):
    top = PANEL_TOP_IN["b"]
    axes = _two_axes(fig, fx, fy, top + AXES_DROP_IN, PANEL_AXES_IN["b"])
    for ax, vec in zip(axes, ("read", "write")):
        ax.axhspan(-rb, rb, color=GS.BAND, lw=0, zorder=0)
        ax.axhline(0, color=GS.BASELINE, lw=0.6, zorder=1)
        for z, (series, (_lab, ck, dashed, alpha)) in enumerate(SERIES_B.items()):
            col = _colour(ck)
            q = s[s.panel.eq("b") & s.space.eq(vec) & s.series.eq(series)].sort_values("layer")
            ax.fill_between(q.layer, q["min"], q["max"], color=col, alpha=alpha, lw=0, zorder=2 + z)
            ax.plot(q.layer, q["mean"], color=col, lw=GS.LW, ls=DASH if dashed else "-", zorder=8 + z)
        ax.text(31.6, -rb - 0.008, BAND_LABEL_B, ha="right", va="top", fontsize=GS.FS_SMALL - 0.8, color=GS.MUTED, zorder=1)
        ax.set_xlim(-0.8, 31.8)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        ax.set_xlabel("Layer", labelpad=1)
        GS.grid(ax, "y")
        GS.facet(ax, VECTOR_TITLE[vec], y=1.08)
    axes[0].set_ylim(-0.16, 1.0)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("Cosine between the two\nobjects' vectors", labelpad=2)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    hs = [(Patch(facecolor=_colour(ck), alpha=alpha, lw=0), Line2D([], [], color=_colour(ck), lw=GS.LW, ls=DASH if dashed else "-"))
          for _lab, ck, dashed, alpha in SERIES_B.values()]
    labels = [lab for lab, *_ in SERIES_B.values()]
    hs.append(Patch(facecolor=GS.BAND, edgecolor="none"))
    labels.append("random vectors, " + BAND_LABEL_B)
    order = [0, 5, 1, 2, 3, 4]                          # the legend fills by column: [CK, random], [CC, KK], [SC, SK]
    fig.legend(handles=[hs[i] for i in order], labels=[labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(fx(LEGEND_X_IN), fy(top)), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.35,
               handlelength=2.0, fontsize=GS.FS_SMALL)
    return axes


def overlap_figure(sc: dict, name: str = FIGURE, out_dirs=None, extra_dirs=()):
    """figA_cK_overlap as the manuscript prints it (appendix_geometry_style; the saved PNG is the manuscript's pixel for pixel): (a) LoRA
    overlaps per layer in the output and input spaces, log scale, with the random-subspace band; (b) ReLU read and write cosines per
    layer with the +-2/sqrt(d) band; a legend row over each panel and the letters a and b. Saved by GS.save into `out_dirs` (default
    R.FIGS) and the mirrors."""
    s, band = summary_frame(sc), band_frame(sc)
    W, H = CANVAS_IN
    fx = lambda x: x / W
    fy = lambda y: 1 - y / H
    with GS.style():
        fig = plt.figure(figsize=(W, H))
        _panel_a(fig, fx, fy, s, band)
        _panel_b(fig, fx, fy, s, random_cosine_band(sc))
        for ch, y in PANEL_TOP_IN.items():
            GS.panel_letter(fig, fx(LETTER_X_IN), fy(y), ch)
        GS.save(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig
