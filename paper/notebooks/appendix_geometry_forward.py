"""appendix_geometry_forward: two judge-free geometry figures of the control-directions appendix, read from the committed sidecar
`data/appendix_geometry_forward.json` (written by `data/build_appendix_geometry_forward.py`): the refusal enrichment (Figure 9's lower
half) and the forward-pass figure, which the paper does not print but whose numbers its text quotes.

    import appendix_geometry_forward as GF
    gf = GF.load_geometry_forward()                    # refuses, naming every problem, unless the file is whole and self-consistent
    GF.refusal_enrichment_figure(gf)                   # figA_refusal_enrichment (Figure 9, bottom)
    GF.displacement_figure(gf)                         # figA_displacement (notebook only)
    GF.table_displacement(gf); GF.table_refusal_enrichment(gf); GF.text_numbers(gf)
    GF.registry_check(gf, objects)                     # a live render: every object id against the registry's object table

THE OBJECTS are the paper's Qwen3.5-4B base-to-instruct fits, seeds 1-3: theta_C (constitution), theta_K (control), -theta_K (theta_K with
its write negated, applied as stored) and theta_C - theta_K (the difference object), as a ReLU steer and as a LoRA; the refusal steer
(Arditi et al., 2024) at c = +0.5 and -0.5 (one run per sign); and the safety positive controls (one ReLU steer, one LoRA, distilled from a
safety-refusal system prompt on the instruct checkpoint). Every object is pinned by sha256, the registry's object_id.

figA_displacement (forward passes, teacher-forced on fixed text):
  (a) mean per-token KL(p_X || p_unsteered) over prompt and reply positions after the first, on attack prompts (T2) and benign prompts
      (T3) with the unsteered model's replies; a large mark is the mean of the three seeds, a small one a single seed.
  (b) the change in the projection of the post-instruction residual state on the unit refusal direction of the same layer, at the
      output of block 28 and the last template token, on the refusal-direction fit's validation prompts (harmful, harmless); the
      refusal steer with its 95% interval over prompts. This REPLACES the first-reply-token refusal score the interim figure drew,
      which sits near its floor on harmless prompts and its ceiling on harmful ones and also rises when a reply opens with "I'd be
      happy". The site is the one the appendix text reads the projection at; `site=` moves it.
  (c) per layer, on the difference object's trajectory and on the unsteered one (same text): the correlation of the two ReLU gates
      over tokens, and kappa_C, the share of theta_C's realized write that theta_K's write removes by projection (ReLU at every layer,
      LoRA at its 8); lines and marks are seeds pooled, bands and bars the seed range.
figA_refusal_enrichment (weights, against 279 refusal directions: every layer but the last, 9 template positions): the share of
  squared norm in each layer's refusal span (left) and on the first refusal principal component (right), divided by what a random
  direction puts there, for the ReLU read and write vectors and the LoRA output space (Delta-W-weighted); large marks the mean of the
  three seeds, small marks single seeds, the safety positive controls one object each (theta_C green, theta_K blue, the safety
  control yellow; circles ReLU, squares LoRA); grey: the random-direction 95% range. One panel, no letter: it is the lower half of a
  figure whose upper half is the sign decomposition (appendix_geometry_evenodd).
  The sidecar also carries the LoRA principal angles of an earlier figure (the overlap figure, appendix_geometry_overlap, replaces
  them) and the signed cosine of each ReLU write with the same-layer refusal direction (positions -5 and -1), which is not drawn:
  `text_numbers` recomputes the layer means the text quotes.

THE STYLE of figA_refusal_enrichment is the manuscript's (`appendix_geometry_style`: its 5.5 x 2.29 in canvas, fonts, palette, marks
and legend rows, saved at the canvas as pdf, a 300 dpi png and svg by `GS.save`), so the png it writes is the manuscript's pixel for
pixel. figA_displacement, which the paper does not print, keeps the notebook's (`R.style()` via `AC.style_context()`, the paper palette
`R.COLOR` and labels `R.LABEL`, the ink colours and the grey band colour `R.RULE`) and is saved by `AC.save_figure`. Both save png, pdf
and svg into `R.FIGS` and any mirror.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import registry_nb as R
import appendix_common as AC
import appendix_geometry_style as GS

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.join(HERE, "data", "appendix_geometry_forward.json")
SCHEMA = "appendix_geometry_forward/1"
BUILDER = "paper/notebooks/data/build_appendix_geometry_forward.py"
FIG_DISPLACEMENT = "figA_displacement"
FIG_ENRICHMENT = "figA_refusal_enrichment"
SITE = (28, -1)                                  # (layer, position): block 28's output, the last template token
SEEDS = ("1", "2", "3")
FAMILIES = ("relu", "lora")
ARMS = ("pK", "mK", "C", "V")
STEER_ARMS = ("pA", "mA")
EVALS = ("arditi_harmful_val", "arditi_harmless_val")
POSITIONS = (-5, -1)
N_LAYERS = 32
LORA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
PAIR_GROUPS = (("C-K", ("C-K same seed", "C-K cross seed")), ("C-C", ("C-C seed ceiling",)), ("K-K", ("K-K seed ceiling",)))
ROLE_OBJECT = {("relu", "C"): "constitution_relu", ("relu", "K"): "control_relu", ("relu", "mK"): "control_relu",
               ("relu", "V"): "difference_relu", ("lora", "C"): "constitution_lora", ("lora", "K"): "control_lora",
               ("lora", "mK"): "control_lora", ("lora", "V"): "difference_lora", ("arditi", "pA"): "arditi_allpos",
               ("arditi", "mA"): "arditi_allpos", ("relu", "S"): "safety_relu", ("lora", "S"): "safety_lora"}
ARM_ROLE = {"pK": "K", "mK": "K", "C": "C", "V": "V"}        # the colour an arm is drawn in (the -theta_K arm: theta_K's, open)
FIT_MODEL = {"C": "qwen35_4b_base", "K": "qwen35_4b_base", "mK": "qwen35_4b_base", "V": "qwen35_4b_base",
             "pA": "qwen35_4b_it", "mA": "qwen35_4b_it", "S": "qwen35_4b_it"}
SHAPE = {"relu": "o", "lora": "s", "arditi": "D"}
TICK = {"pK": r"$+\theta_K$", "mK": r"$-\theta_K$", "C": r"$\theta_C$", "V": r"$\theta_C-\theta_K$",
        "pA": "refusal\n$+0.5$", "mA": "refusal\n$-0.5$"}
GREEK = {"K": r"$\theta_K$", "C": r"$\theta_C$", "V": r"$\theta_C-\theta_K$"}
COMPONENTS = (("relu", "read", "ReLU read\n(gate vectors)"), ("relu", "write", "ReLU write\n(directions)"),
              ("lora", "write_post", "LoRA output\n(" + r"$\Delta W$" + "-weighted)"))
MEAN_S, SEED_S, STEER_S = 190.0, 46.0, 170.0      # marker areas (pt^2): a seed mean, one seed, the refusal steer
FAM_DX, SEED_DX = {"relu": -0.19, "lora": 0.19}, 0.085
WIDTH = AC.MODEL_WIDTH                            # 16.4 in, the comparison figures' canvas, so text prints at their size
HEX = re.compile(r"^[0-9a-f]{64}$")
COLUMNS = {
    ("displacement", "kl"): ["family", "set", "arm", "seed", "mean_kl", "n_tokens"],
    ("displacement", "projection", "unsteered"): ["eval", "position", "layer", "mean_proj"],
    ("displacement", "kl_per_eval"): ["family", "seed", "set", "eval", "arm", "kl", "first_token_kl", "n_items", "n_tokens"],
    ("displacement", "projection", "delta"): ["family", "seed", "eval", "position", "layer"] + list(ARMS),
    ("displacement", "projection", "refusal_steer"): ["eval", "position", "layer", "arm", "delta", "se", "n_prompts"],
    ("displacement", "writes", "relu"): ["traj", "layer", "corr_g", "corr_g_seed_min", "corr_g_seed_max", "kappa_C",
                                         "kappa_C_seed_min", "kappa_C_seed_max"],
    ("displacement", "writes", "lora"): ["traj", "layer", "kappa_C", "kappa_C_seed_min", "kappa_C_seed_max"],
    ("displacement", "writes", "pooled_over_layers"): ["family", "traj", "stat", "value"],
    ("refusal_subspace", "enrichment"): ["family", "component", "which", "obj", "seed", "enrichment"],
    ("refusal_subspace", "enrichment_null"): ["family", "component", "which", "obj", "expected_isotropic", "null_q0.025", "null_q0.975",
                                              "p_null_max", "n_seeds"],
    ("refusal_subspace", "signed_cosine"): ["obj", "seed", "layer", "cos_pos-5", "cos_pos-1"],
    ("refusal_subspace", "principal_angles"): ["pair_type", "x", "y", "layer", "out_top", "in_top", "cos_F"],
    ("refusal_subspace", "frobenius_stacked"): ["pair_type", "x", "y", "cos_F"],
}


class GeometryForwardSidecarError(ValueError):
    pass


# ---- reading and checking the sidecar -----------------------------------------------------------------------------------------------
def _get(sc: dict, path: tuple):
    x = sc
    for k in path:
        x = x[k]
    return x


def frame(sc: dict, *path) -> pd.DataFrame:
    """One of the sidecar's tables (`{"columns", "rows"}`) as a DataFrame, e.g. frame(gf, "displacement", "kl")."""
    b = _get(sc, path)
    return pd.DataFrame(b["rows"], columns=b["columns"])


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def geometry_forward_problems(sc: dict) -> list:
    """Every reason the sidecar cannot be drawn from, as sentences; empty when it is whole. Checks the stamp, the provenance, the
    object pins and their registry claims, each table's shape and coverage, and that every pooled number a panel draws is the mean of
    its seeds."""
    p = []
    if sc.get("schema") != SCHEMA:
        return [f"schema {sc.get('schema')!r}, expected {SCHEMA!r}"]
    if (sc.get("judge") or {}).get("model") != "none":
        p.append(f"judge stamp {sc.get('judge')!r}: these figures are judge-free and the stamp must say model 'none'")
    pv = sc.get("provenance") or {}
    for k in ("generated_utc", "script", "script_sha256", "source_files", "gpu_run", "registry"):
        if k not in pv:
            p.append(f"provenance lacks {k!r}")
    if pv.get("script") != BUILDER:
        p.append(f"provenance.script {pv.get('script')!r}, expected {BUILDER!r}")
    files = pv.get("source_files") or []
    if not files or any(not HEX.match(str(f.get("sha256", ""))) or not f.get("name") for f in files):
        p.append("provenance.source_files must name every source with its sha256")
    leaks = sorted({s for s in _strings(sc) if s.startswith(("/", "~")) or "/home/" in s})
    if leaks:
        p.append(f"absolute or home paths in the sidecar: {leaks[:3]}")
    # objects
    objs = sc.get("objects") or []
    ids = [o.get("object_id", "") for o in objs]
    if len(set(ids)) != len(ids) or not all(HEX.match(i) for i in ids):
        p.append("objects: every object_id must be a distinct sha256")
    have = {(o["family"], o["role"], o["seed"]) for o in objs}
    want = ({(f, r, s) for f in FAMILIES for r in ("C", "K", "mK", "V") for s in SEEDS}
            | {("arditi", "pA", ""), ("arditi", "mA", ""), ("relu", "S", ""), ("lora", "S", "")})
    if want - have:
        p.append(f"objects: missing {sorted(want - have)}")
    for o in objs:
        reg = o.get("registry") or {}
        lab = ROLE_OBJECT.get((o["family"], o["role"]))
        if reg.get("paper_label") != lab or reg.get("fit_model_id") != FIT_MODEL.get(o["role"]):
            p.append(f"object {o['object_id'][:12]} ({o['family']} {o['role']} s{o['seed']}): registry claim "
                     f"{reg.get('paper_label')}/{reg.get('fit_model_id')}, expected {lab}/{FIT_MODEL.get(o['role'])}")
    # tables
    for path, cols in COLUMNS.items():
        try:
            b = _get(sc, path)
        except (KeyError, TypeError):
            p.append(f"missing table {'.'.join(path)}")
            continue
        if b.get("columns") != cols or any(len(r) != len(cols) for r in b.get("rows", [])) or not b.get("rows"):
            p.append(f"table {'.'.join(path)}: columns {b.get('columns')}, expected {cols}, with rows of that width")
    if p:
        return p
    kl = frame(sc, "displacement", "kl")
    for (fam, st, arm), g in kl[kl.family.isin(FAMILIES)].groupby(["family", "set", "arm"]):
        g = g.set_index("seed")
        if set(g.index) != set(SEEDS) | {"pooled"}:
            p.append(f"kl {fam} {st} {arm}: seeds {sorted(g.index)}")
        elif not math.isclose(g.loc[list(SEEDS), "mean_kl"].mean(), g.at["pooled", "mean_kl"], rel_tol=1e-9):
            p.append(f"kl {fam} {st} {arm}: the pooled value is not the mean of its seeds")
    for fam in FAMILIES:
        for st in ("T2", "T3"):
            for arm in ARMS:
                if not len(kl[kl.family.eq(fam) & kl.set.eq(st) & kl.arm.eq(arm)]):
                    p.append(f"kl: no {fam} {st} {arm} rows")
    pe = frame(sc, "displacement", "kl_per_eval")
    pe = pe.assign(s=pe.kl * pe.n_tokens).groupby(["family", "seed", "set", "arm"]).agg(s=("s", "sum"), n=("n_tokens", "sum"))
    k = kl.set_index(["family", "seed", "set", "arm"]).mean_kl
    for key, r in pe.iterrows():
        if key not in k.index or not math.isclose(r.s / r.n, k[key], rel_tol=1e-9):
            p.append(f"kl_per_eval {key}: the token-weighted KL over evaluations is not the text set's KL")
            break
    d = frame(sc, "displacement", "projection", "delta")
    keys = d[["family", "seed", "eval", "position", "layer"]].apply(tuple, axis=1)
    want = {(f, s, e, q, l) for f in FAMILIES for s in SEEDS for e in EVALS for q in POSITIONS for l in range(N_LAYERS)}
    if keys.duplicated().any() or set(keys) != want or d[list(ARMS)].isna().any().any():
        p.append("projection.delta must hold every family x seed x prompt set x position x layer once, with no missing arm")
    u = frame(sc, "displacement", "projection", "unsteered")
    if set(u[["eval", "position", "layer"]].apply(tuple, axis=1)) != {(e, q, l) for e in EVALS for q in POSITIONS for l in range(N_LAYERS)}:
        p.append("projection.unsteered must hold every prompt set x position x layer")
    st_ = frame(sc, "displacement", "projection", "refusal_steer")
    if (set(st_[["eval", "position", "layer", "arm"]].apply(tuple, axis=1))
            != {(e, q, l, a) for e in EVALS for q in POSITIONS for l in range(N_LAYERS) for a in STEER_ARMS}) or st_.se.isna().any():
        p.append("projection.refusal_steer must hold every prompt set x position x layer x sign, each with its SE")
    rw, lw = frame(sc, "displacement", "writes", "relu"), frame(sc, "displacement", "writes", "lora")
    for traj in ("V", "clean"):
        if sorted(rw[rw.traj.eq(traj)].layer) != list(range(N_LAYERS)):
            p.append(f"writes.relu {traj}: layers {sorted(rw[rw.traj.eq(traj)].layer)}")
        if sorted(lw[lw.traj.eq(traj)].layer) != list(LORA_LAYERS):
            p.append(f"writes.lora {traj}: layers {sorted(lw[lw.traj.eq(traj)].layer)}")
    en = frame(sc, "refusal_subspace", "enrichment")
    for (fam, comp, which, obj), g in en.groupby(["family", "component", "which", "obj"]):
        if sorted(g.seed) != ([""] if obj == "S" else list(SEEDS)):
            p.append(f"enrichment {fam} {comp} {which} {obj}: seeds {sorted(g.seed)}")
    if len(en.groupby(["family", "component", "which", "obj"])) != len(COMPONENTS) * 2 * 3:
        p.append("enrichment: every component x measure x object (C, K, S) must be present")
    nl = frame(sc, "refusal_subspace", "enrichment_null")
    if len(nl) != len(COMPONENTS) * 2 * 3:
        p.append("enrichment_null: one row per component x measure x object")
    cs = frame(sc, "refusal_subspace", "signed_cosine")
    for obj, seeds in (("K", SEEDS), ("C", SEEDS), ("S", ("",))):
        g = cs[cs.obj.eq(obj)]
        if sorted(g.seed.unique()) != sorted(seeds) or (g.groupby("seed").size() != N_LAYERS - 1).any():
            p.append(f"signed_cosine {obj}: every seed at layers 0-30")
    pa = frame(sc, "refusal_subspace", "principal_angles")
    for _, types in PAIR_GROUPS:
        g = pa[pa.pair_type.isin(types)]
        if not len(g) or sorted(g.layer.unique()) != list(LORA_LAYERS):
            p.append(f"principal_angles {types}: every LoRA layer")
    for k in ("out_2560", "in_4096"):
        if not all(q in (sc["refusal_subspace"].get("haar_null") or {}).get(k, {}) for q in ("q0.025", "q0.975")):
            p.append(f"haar_null.{k} lacks its 95% quantiles")
    return p


def load_geometry_forward(path: str = SIDECAR) -> dict:
    """The sidecar at `path`, after `geometry_forward_problems`; raises GeometryForwardSidecarError naming every problem."""
    with open(path, encoding="utf-8") as fh:
        sc = json.load(fh)
    problems = geometry_forward_problems(sc)
    if problems:
        raise GeometryForwardSidecarError(f"{os.path.basename(path)}: " + "; ".join(problems))
    return sc


def registry_check(sc: dict, objects: pd.DataFrame) -> pd.DataFrame:
    """One row per sidecar object with `known`: the registry's object table (registry_nb.OBJECTS_SQL) holds that id with the paper
    label, fit checkpoint and derivation the sidecar claims, a fitted theta_C or theta_K under its seed, and a derived object (-theta_K,
    the difference objects, the -0.5 refusal steer) with the parents the sidecar names."""
    o = objects.set_index("object_id")
    rows = []
    for s in sc["objects"]:
        reg, ok, why = s["registry"], s["object_id"] in o.index, ""
        if ok:
            r = o.loc[s["object_id"]]
            rep = r.get("fit_replicate_label")
            rep = None if rep is None or (isinstance(rep, float) and math.isnan(rep)) else str(rep)
            parents = json.loads(r["parent_object_ids_json"]) if isinstance(r.get("parent_object_ids_json"), str) else []
            checks = {"paper_label": str(r["paper_label"]) == reg["paper_label"], "fit_model_id": str(r["fit_model_id"]) == reg["fit_model_id"],
                      "derivation": str(r.get("derivation")) == str(reg["derivation"]), "replicate": rep == reg["fit_replicate_label"],
                      "parents": parents == list(reg.get("parents") or [])}
            ok = all(checks.values())
            why = ", ".join(k for k, v in checks.items() if not v)
        else:
            why = "not in the registry"
        rows.append(dict(object_id=s["object_id"], family=s["family"], role=s["role"], seed=s["seed"], paper_label=reg["paper_label"],
                         known=bool(ok), mismatch=why))
    return pd.DataFrame(rows)


# ---- the panels' numbers ------------------------------------------------------------------------------------------------------------
def kl_frame(sc: dict) -> pd.DataFrame:
    """Panel (a): one row per family x text set x arm x seed ('1'-'3', and 'mean', the mean of the three)."""
    kl = frame(sc, "displacement", "kl")
    kl = kl[kl.family.isin(FAMILIES) & kl.set.isin(["T2", "T3"])].copy()
    kl["seed"] = kl.seed.replace({"pooled": "mean"})
    return kl.reset_index(drop=True)


def unsteered_level(sc: dict, eval_id: str, site: tuple = SITE) -> float:
    u = frame(sc, "displacement", "projection", "unsteered")
    layer, pos = site
    return float(u[u["eval"].eq(eval_id) & u.position.eq(pos) & u.layer.eq(layer)].mean_proj.iat[0])


def projection_frame(sc: dict, site: tuple = SITE) -> pd.DataFrame:
    """Panel (b) at `site` (layer, position): one row per family x seed ('1'-'3', 'mean') x prompt set x arm with the change in the
    projection, and the refusal steer (family 'arditi', no seed) with its SE over prompts."""
    layer, pos = site
    d = frame(sc, "displacement", "projection", "delta")
    d = d[d.layer.eq(layer) & d.position.eq(pos)]
    long = d.melt(id_vars=["family", "seed", "eval"], value_vars=list(ARMS), var_name="arm", value_name="delta")
    mean = long.groupby(["family", "eval", "arm"], as_index=False).delta.mean().assign(seed="mean")
    st = frame(sc, "displacement", "projection", "refusal_steer")
    st = st[st.layer.eq(layer) & st.position.eq(pos)].assign(family="arditi", seed="")[["family", "seed", "eval", "arm", "delta", "se"]]
    out = pd.concat([long.assign(se=np.nan), mean.assign(se=np.nan), st], ignore_index=True)
    return out[["family", "seed", "eval", "arm", "delta", "se"]]


def odd_fraction_frame(sc: dict, position: int = -5, layers=range(15, 31)) -> pd.DataFrame:
    """The sign-dependent part of theta_K's change, (D(+theta_K) - D(-theta_K)) / 2, and the refusal steer's, (D(+0.5) - D(-0.5)) / 2,
    on harmful prompts, as a fraction of the unsteered harmful-minus-harmless gap at the same layer and position: one row per layer and
    family x seed (the refusal steer: family 'arditi')."""
    u = frame(sc, "displacement", "projection", "unsteered")
    u = u[u.position.eq(position)].pivot_table(index="layer", columns="eval", values="mean_proj")
    gap = u["arditi_harmful_val"] - u["arditi_harmless_val"]
    d = frame(sc, "displacement", "projection", "delta")
    d = d[d.position.eq(position) & d["eval"].eq("arditi_harmful_val") & d.layer.isin(list(layers))].copy()
    d["odd_frac"] = (d.pK - d.mK) / 2 / d.layer.map(gap)
    st = frame(sc, "displacement", "projection", "refusal_steer")
    st = st[st.position.eq(position) & st["eval"].eq("arditi_harmful_val") & st.layer.isin(list(layers))]
    st = st.pivot_table(index="layer", columns="arm", values="delta")
    a = pd.DataFrame({"family": "arditi", "seed": "", "layer": st.index, "odd_frac": ((st.pA - st.mA) / 2 / st.index.map(gap)).values})
    return pd.concat([d[["family", "seed", "layer", "odd_frac"]], a], ignore_index=True)


def writes_frames(sc: dict) -> tuple:
    return frame(sc, "displacement", "writes", "relu"), frame(sc, "displacement", "writes", "lora")


def enrichment_frame(sc: dict) -> pd.DataFrame:
    """The refusal enrichment figure's marks: one row per component x measure ('span', 'pc1') x object (C, K, S) x seed ('1'-'3',
    'mean'; the safety control is one object, seed '')."""
    en = frame(sc, "refusal_subspace", "enrichment")
    mean = en[en.obj.ne("S")].groupby(["family", "component", "which", "obj"], as_index=False).enrichment.mean().assign(seed="mean")
    return pd.concat([en, mean], ignore_index=True)


def enrichment_band(sc: dict) -> pd.DataFrame:
    """The grey random-direction 95% range per component and measure: the widest range over the three objects' nulls, each null's
    quantiles divided by the share a random direction is expected to carry."""
    nl = frame(sc, "refusal_subspace", "enrichment_null")
    nl = nl.assign(lo=nl["null_q0.025"] / nl.expected_isotropic, hi=nl["null_q0.975"] / nl.expected_isotropic)
    return nl.groupby(["family", "component", "which"], as_index=False).agg(lo=("lo", "min"), hi=("hi", "max"))


def signed_cosine_frame(sc: dict) -> pd.DataFrame:
    return frame(sc, "refusal_subspace", "signed_cosine")


# ---- tables --------------------------------------------------------------------------------------------------------------------------
def table_displacement(sc: dict, site: tuple = SITE) -> pd.DataFrame:
    """The numbers panels (a) and (b) draw: per panel, family and arm, each seed's value and the mean of the three."""
    rows = []
    kl = kl_frame(sc)
    for st, lab in (("T2", "(a) KL, attack texts"), ("T3", "(a) KL, benign texts")):
        for fam in FAMILIES:
            for arm in ARMS:
                g = kl[kl.set.eq(st) & kl.family.eq(fam) & kl.arm.eq(arm)].set_index("seed").mean_kl
                rows.append(dict(panel=lab, family=fam, arm=arm, **{f"s{s}": g[s] for s in SEEDS}, mean=g["mean"], se=np.nan))
    pf = projection_frame(sc, site)
    for ev, lab in zip(EVALS, ("(b) projection change, harmful", "(b) projection change, harmless")):
        for arm in STEER_ARMS:
            r = pf[pf.family.eq("arditi") & pf["eval"].eq(ev) & pf.arm.eq(arm)].iloc[0]
            rows.append(dict(panel=lab, family="arditi", arm=arm, s1=np.nan, s2=np.nan, s3=np.nan, mean=r.delta, se=r.se))
        for fam in FAMILIES:
            for arm in ARMS:
                g = pf[pf.family.eq(fam) & pf["eval"].eq(ev) & pf.arm.eq(arm)].set_index("seed").delta
                rows.append(dict(panel=lab, family=fam, arm=arm, **{f"s{s}": g[s] for s in SEEDS}, mean=g["mean"], se=np.nan))
    return pd.DataFrame(rows)


def table_refusal_enrichment(sc: dict) -> pd.DataFrame:
    """The refusal enrichment figure's numbers: per component, measure and object, each seed's enrichment, the seed mean, the Haar p (largest over seeds)
    and the random-direction band."""
    en, band = enrichment_frame(sc), enrichment_band(sc).set_index(["family", "component", "which"])
    nl = frame(sc, "refusal_subspace", "enrichment_null").set_index(["family", "component", "which", "obj"])
    rows = []
    for fam, comp, _ in COMPONENTS:
        for which in ("span", "pc1"):
            for obj in ("C", "K", "S"):
                g = en[en.family.eq(fam) & en.component.eq(comp) & en.which.eq(which) & en.obj.eq(obj)].set_index("seed").enrichment
                rows.append(dict(component=f"{fam} {comp}", measure=which, obj=obj, **{f"s{s}": g.get(s, np.nan) for s in SEEDS},
                                 mean=g["mean"] if obj != "S" else g[""], haar_p_max=nl.at[(fam, comp, which, obj), "p_null_max"],
                                 band_lo=band.at[(fam, comp, which), "lo"], band_hi=band.at[(fam, comp, which), "hi"]))
    return pd.DataFrame(rows)


def text_numbers(sc: dict, site: tuple = SITE) -> pd.DataFrame:
    """The numbers the appendix text quotes from this sidecar (the forward passes, the enrichment, the signed cosines, the LoRA
    Frobenius cosines), recomputed from it: key, value and how."""
    out = []

    def put(key, value, how):
        out.append(dict(key=key, value=value, how=how))

    kl = kl_frame(sc)
    k = kl[kl.seed.eq("mean")].set_index(["family", "set", "arm"]).mean_kl
    for fam in FAMILIES:
        put(f"kl_{fam}_T2_V", k[(fam, "T2", "V")], "mean per-token KL, attack texts, theta_C - theta_K, seed mean")
        put(f"kl_{fam}_T2_C", k[(fam, "T2", "C")], "same, theta_C")
    per = kl[kl.seed.isin(SEEDS)].pivot_table(index=["family", "set", "seed"], columns="arm", values="mean_kl")
    put("kl_V_below_C_cells", f"{int((per.V < per.C).sum())}/{len(per)}", "family x text set x seed cells where theta_C - theta_K's KL is below theta_C's")
    pe = frame(sc, "displacement", "kl_per_eval")
    pv = pe[pe.arm.isin(["C", "V"])].pivot_table(index=["family", "seed", "eval"], columns="arm", values="kl")
    put("kl_V_below_C_eval_cells", f"{int((pv.V < pv.C).sum())}/{len(pv)}",
        "family x seed x evaluation cells (attack and benign texts) where theta_C - theta_K's KL is below theta_C's")
    for fam in FAMILIES:
        q = pe[pe.family.eq(fam) & pe.arm.isin(["C", "V"])]
        w = {a: float(np.average(g.first_token_kl, weights=g.n_items)) for a, g in q.groupby("arm")}
        ft = q.pivot_table(index=["seed", "eval"], columns="arm", values="first_token_kl")
        put(f"first_token_kl_{fam}", (round(w["V"], 3), round(w["C"], 3), f"{int((ft.V > ft.C).sum())}/{len(ft)}"),
            "first-reply-token KL, item-weighted over seeds and evaluations: (theta_C - theta_K, theta_C, cells where the former is larger)")
    t1 = frame(sc, "displacement", "kl")
    t1 = t1[t1.set.eq("T1")]
    st = t1[t1.family.eq("arditi")].mean_kl
    arms = t1[t1.family.isin(FAMILIES) & t1.seed.eq("pooled")].mean_kl
    put("kl_T1_refusal_steer", f"{st.min():.3f}-{st.max():.3f}", "validation prompts, refusal steer at c = +-0.5")
    put("kl_T1_objects", f"{arms.min():.2f}-{arms.max():.2f}", "validation prompts, the objects' four arms, seed means")
    of = odd_fraction_frame(sc)
    for fam in FAMILIES:
        g = of[of.family.eq(fam)]
        m = g.groupby("layer").odd_frac.mean()
        put(f"odd_frac_{fam}_seed_mean_range", (round(m.min(), 3), round(m.max(), 3)),
            "sign-dependent part of theta_K's change, harmful prompts, position -5, layers 15-30, / unsteered harmful-harmless gap; range of seed means")
        put(f"odd_frac_{fam}_per_seed_range", {s: (round(g[g.seed.eq(s)].odd_frac.min(), 3), round(g[g.seed.eq(s)].odd_frac.max(), 3)) for s in SEEDS},
            "same, per seed")
    a = of[of.family.eq("arditi")].odd_frac
    put("odd_frac_refusal_steer_range", (round(a.min(), 3), round(a.max(), 3)), "same, the refusal steer at c = +-0.5")
    layer, pos = site
    P = unsteered_level(sc, "arditi_harmful_val", site)
    d = frame(sc, "displacement", "projection", "delta")
    d = d[d.layer.eq(layer) & d.position.eq(pos) & d["eval"].eq("arditi_harmful_val")]
    put("unsteered_harmful_projection", P, f"unsteered mean projection on harmful prompts at layer {layer}, position {pos}")
    for fam in FAMILIES:
        g = d[d.family.eq(fam)]
        drop = lambda c: -100 * g[c] / P
        put(f"drop_{fam}_pmK_pct", (round(min(drop("pK").min(), drop("mK").min()), 2), round(max(drop("pK").max(), drop("mK").max()), 2)),
            "drop of the harmful-prompt projection under +theta_K and -theta_K, % of the unsteered projection, over seeds")
        put(f"drop_{fam}_C_pct", (round(drop("C").min(), 2), round(drop("C").max(), 2)), "same under theta_C")
        put(f"drop_{fam}_V_pct", (round(drop("V").min(), 2), round(drop("V").max(), 2)), "same under theta_C - theta_K")
    po = frame(sc, "displacement", "writes", "pooled_over_layers").set_index(["family", "traj", "stat"]).value
    put("gate_corr_relu_V", po[("relu", "V", "corr_g")], "ReLU gate correlation over tokens on the difference trajectory, layers and seeds pooled")
    put("kappa_C_relu_V", po[("relu", "V", "kappa_C")], "kappa_C, ReLU, same")
    put("kappa_C_lora_V", po[("lora", "V", "kappa_C")], "kappa_C, LoRA, same")
    em = enrichment_frame(sc)
    pc1 = em[em.which.eq("pc1") & (em.seed.eq("mean") | em.obj.eq("S"))]
    put("enrich_pc1_K_range", (round(pc1[pc1.obj.eq("K")].enrichment.min(), 2), round(pc1[pc1.obj.eq("K")].enrichment.max(), 2)),
        "control objects on the first refusal PC, seed means over the three components")
    put("enrich_pc1_S_range", (round(pc1[pc1.obj.eq("S")].enrichment.min(), 2), round(pc1[pc1.obj.eq("S")].enrichment.max(), 2)),
        "safety positive controls, same")
    put("enrich_pc1_C_max", round(pc1[pc1.obj.eq("C")].enrichment.max(), 2), "constitution objects, largest")
    sp = em[em.which.eq("span") & em.component.eq("write") & em.family.eq("relu")]
    put("enrich_span_K_write", round(sp[sp.obj.eq("K") & sp.seed.eq("mean")].enrichment.iat[0], 2), "control ReLU write, per-layer span, seed mean")
    put("enrich_span_S_write", round(sp[sp.obj.eq("S")].enrichment.iat[0], 2), "safety ReLU write, same")
    nl = frame(sc, "refusal_subspace", "enrichment_null").set_index(["family", "component", "which", "obj"])
    put("haar_p_K_write_span", nl.at[("relu", "write", "span", "K"), "p_null_max"], "Haar p of the control ReLU write's span share, largest over seeds")
    cs = signed_cosine_frame(sc)
    for obj in ("K", "S"):
        for q in (-5, -1):
            m = cs[cs.obj.eq(obj)].groupby("layer")[f"cos_pos{q}"].mean().mean()
            put(f"signed_cos_{obj}_pos{q}", round(m, 4), f"{obj} ReLU write vs the same-layer refusal direction, mean over layers of the seed mean")
    fs = frame(sc, "refusal_subspace", "frobenius_stacked")
    ck = fs[fs.pair_type.isin(["C-K same seed", "C-K cross seed"])].cos_F
    put("frobenius_CK_range", (round(ck.min(), 3), round(ck.max(), 3)), "Frobenius cosine of theta_C and theta_K's LoRA updates, stacked over the 8 layers")
    put("n_refusal_directions", sc["refusal_subspace"]["directions"]["n_vectors"], "refusal directions in the subspace")
    return pd.DataFrame(out)


def provenance_line(sc: dict) -> str:
    pv = sc["provenance"]
    return (f"{os.path.basename(SIDECAR)}: forward-pass run {pv['gpu_run']['name']} (code {pv['gpu_run']['code_sha'][:9]}, "
            f"{len(pv['gpu_run']['sanity_checks_passed'])} sanity checks passed), weight analyses against refusal directions "
            f"{pv['weights']['refusal_directions_sha256'][:8]}; {len(sc['objects'])} objects pinned by sha256 and bound to the registry "
            f"(head {pv['registry']['head_at_build']}); {len(pv['source_files'])} source files; judge: none (judge-free); "
            f"built {pv['generated_utc']} by {pv['script']}")


# ---- drawing helpers -----------------------------------------------------------------------------------------------------------------
def _color(family: str, role: str) -> str:
    return R.COLOR[ROLE_OBJECT[(family, role)]]


def _mark(ax, x, y, color, marker, s, open_=False, z=4):
    if open_:
        ax.scatter([x], [y], marker=marker, s=s, facecolors="white", edgecolors=color, linewidths=2.2 if s > 80 else 1.6, zorder=z)
    else:
        ax.scatter([x], [y], marker=marker, s=s, color=color, edgecolors="white", linewidths=1.0 if s > 80 else 0.6, zorder=z)


def _seed_marks(ax, x, vals, color, marker, open_=False):
    for v in vals:
        _mark(ax, x + SEED_DX, v, color, marker, SEED_S, open_, z=3)


def _grid(ax):
    ax.grid(True, axis="y", color=R.RULE, lw=0.9)
    ax.set_axisbelow(True)


def _finish(fig, letters, top_groups):
    """The stacked legend above the canvas (`R.legend_stack`, which the tight bounding box keeps), then each panel letter at the left
    edge of the figure, level with its row's titles (placed after the layout has run, so it never meets a legend or a label)."""
    R.legend_stack(fig, top_groups)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x0 = min(inv.transform((ax.get_tightbbox(r).x0, 0))[0] for ax, _ in letters)
    for ax, ch in letters:
        b = ax.title.get_window_extent(r)
        y = inv.transform((0, (b.y0 + b.y1) / 2))[1]
        fig.text(x0 - 0.004, y, ch, fontsize=26, fontweight="bold", color=R.INK, ha="right", va="center", transform=fig.transFigure)


def _handle(color, marker, label, open_=False, ms=11):
    return Line2D([], [], ls="", marker=marker, ms=ms, markerfacecolor="white" if open_ else color,
                  markeredgecolor=color if open_ else "white", markeredgewidth=2.0 if open_ else 1.0, label=label)


def _object_label(family: str, role: str) -> str:
    return f"{R.LABEL[ROLE_OBJECT[(family, role)]]} ({GREEK[role]})"


def _legend(sf, handles, ncol):
    leg = sf.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False, handletextpad=0.35, columnspacing=1.3,
                    borderaxespad=0.2, labelspacing=0.35)
    return leg


# ---- figA_displacement ---------------------------------------------------------------------------------------------------------------
def displacement_figure(sc: dict, name: str = FIG_DISPLACEMENT, site: tuple = SITE, out_dirs=None, extra_dirs=(), height: float = 14.5):
    """figA_displacement in the paper's style: (a) KL from the unsteered model, (b) the change in the refusal-direction projection at
    `site`, (c) the realized-write geometry per layer. Saved by AC.save_figure into `out_dirs` (default R.FIGS) and the mirrors."""
    layer, pos = site
    with AC.style_context():
        fig = plt.figure(figsize=(WIDTH, height), layout="constrained")
        sfa, sfb, sfc = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.05], hspace=0.04)
        # ---- legend (a, b): above the canvas -----------------------------------------------------------------------------------
        objs = [_handle(_color(f, r), SHAPE[f], _object_label(f, r)) for r in ("K", "C", "V") for f in FAMILIES]
        objs += [_handle(R.COLOR["arditi_allpos"], SHAPE["arditi"], f"{R.LABEL['arditi_allpos']}, $c = \\pm 0.5$"),
                 Line2D([], [], color=R.COLOR["arditi_allpos"], lw=2.4, label="its 95% CI over prompts")]
        glyphs = [_handle(R.INK2, "o", "large: mean of 3 seeds"), _handle(R.INK2, "o", "small: one seed", ms=6.5),
                  _handle(R.INK2, "o", r"open: write negated ($-\theta_K$; refusal steer $-0.5$)", open_=True)]
        # ---- (a) KL -----------------------------------------------------------------------------------------------------------
        kl = kl_frame(sc)
        n_texts = sc["texts"]
        axa = sfa.subplots(1, 2, sharey=True)
        for ax, st, lab in ((axa[0], "T2", f"Attack prompts, unsteered replies ({n_texts['T2']})"),
                            (axa[1], "T3", f"Benign prompts, unsteered replies ({n_texts['T3']})")):
            _grid(ax)
            for i, arm in enumerate(ARMS):
                for fam in FAMILIES:
                    g = kl[kl.family.eq(fam) & kl.set.eq(st) & kl.arm.eq(arm)].set_index("seed").mean_kl
                    x, col = i + FAM_DX[fam], _color(fam, ARM_ROLE[arm])
                    _mark(ax, x, g["mean"], col, SHAPE[fam], MEAN_S, open_=arm == "mK")
                    _seed_marks(ax, x, [g[s] for s in SEEDS], col, SHAPE[fam], open_=arm == "mK")
            ax.set_xticks(range(len(ARMS)), [TICK[a] for a in ARMS])
            ax.set_xlim(-0.55, len(ARMS) - 0.45)
            ax.tick_params(axis="x", length=0)
            ax.set_title(lab)
        axa[0].set_ylim(0, None)
        axa[0].set_ylabel("KL from unsteered\n(nats per token)")
        # ---- (b) refusal-direction projection --------------------------------------------------------------------------------
        pf = projection_frame(sc, site)
        XB = {"pA": 0.0, "mA": 1.0, "pK": 2.35, "mK": 3.35, "C": 4.35, "V": 5.35}
        axb = sfb.subplots(1, 2, sharey=True)
        for ax, ev, lab in ((axb[0], "arditi_harmful_val", "Harmful prompts"), (axb[1], "arditi_harmless_val", "Harmless prompts")):
            _grid(ax)
            ax.axhline(0, color=R.INK2, lw=1.0, zorder=1)
            n = sc["displacement"]["projection"]["texts"][ev]
            for arm in STEER_ARMS:
                r = pf[pf.family.eq("arditi") & pf["eval"].eq(ev) & pf.arm.eq(arm)].iloc[0]
                ax.vlines(XB[arm], r.delta - 1.96 * r.se, r.delta + 1.96 * r.se, color=R.COLOR["arditi_allpos"], lw=2.4, zorder=2)
                _mark(ax, XB[arm], r.delta, R.COLOR["arditi_allpos"], SHAPE["arditi"], STEER_S, open_=arm == "mA")
            for arm in ARMS:
                for fam in FAMILIES:
                    g = pf[pf.family.eq(fam) & pf["eval"].eq(ev) & pf.arm.eq(arm)].set_index("seed").delta
                    x, col = XB[arm] + FAM_DX[fam], _color(fam, ARM_ROLE[arm])
                    _mark(ax, x, g["mean"], col, SHAPE[fam], MEAN_S, open_=arm == "mK")
                    _seed_marks(ax, x, [g[s] for s in SEEDS], col, SHAPE[fam], open_=arm == "mK")
            ax.axvline(1.67, color=R.MUTED, lw=1.0, zorder=0)
            ax.set_xticks(list(XB.values()), [TICK[k] for k in XB], linespacing=1.0)
            ax.set_xlim(-0.55, 5.9)
            ax.tick_params(axis="x", length=0)
            ax.set_title(f"{lab} ({n}); unsteered {unsteered_level(sc, ev, site):+.1f}".replace("-", "−"))
        axb[0].set_ylabel(f"Change in refusal-direction\nprojection (layer {layer}, last token)" if pos == -1 else
                          f"Change in refusal-direction\nprojection (layer {layer}, position {pos})")
        # ---- (c) realized-write geometry ---------------------------------------------------------------------------------------
        rw, lw = writes_frames(sc)
        axC, axK = sfc.subplots(1, 2)
        traj_col = {"V": (R.COLOR["difference_relu"], R.COLOR["difference_lora"]), "clean": (R.COLOR["none"], R.COLOR["none"])}
        for traj in ("clean", "V"):
            rc, lc = traj_col[traj]
            z = 3 if traj == "V" else 2
            g = rw[rw.traj.eq(traj)].sort_values("layer")
            axC.fill_between(g.layer, g.corr_g_seed_min, g.corr_g_seed_max, color=rc, alpha=0.18, lw=0, zorder=z - 1)
            axC.plot(g.layer, g.corr_g, color=rc, lw=2.6, zorder=z)
            axK.fill_between(g.layer, g.kappa_C_seed_min, g.kappa_C_seed_max, color=rc, alpha=0.18, lw=0, zorder=z - 1)
            axK.plot(g.layer, g.kappa_C, color=rc, lw=2.6, zorder=z)
            gl = lw[lw.traj.eq(traj)].sort_values("layer")
            off = 0.4 if traj == "V" else -0.4
            axK.vlines(gl.layer + off, gl.kappa_C_seed_min, gl.kappa_C_seed_max, color=lc, lw=2.0, zorder=z + 1)
            axK.scatter(gl.layer + off, gl.kappa_C, marker="s", s=120, color=lc, edgecolors="white", linewidths=1.0, zorder=z + 2)
        for ax in (axC, axK):
            _grid(ax)
            ax.axhline(0, color=R.INK2, lw=1.0, zorder=1)
            ax.set_xlim(-0.8, 31.8)
            ax.set_xticks([0, 8, 16, 24, 31])
            ax.set_xlabel("Layer")
        axC.set_ylabel("Gate correlation,\n" + r"corr$(g_C, g_K)$ over tokens")
        axK.set_ylabel(r"Share of $\theta_C$'s write" + "\n" + r"removed by $\theta_K$'s, $\kappa_C$")
        axC.set_title("ReLU steer")
        axK.set_title("ReLU steer and LoRA")
        hc = [Line2D([], [], color=R.COLOR["difference_relu"], lw=2.6, label=r"ReLU steer, on the $\theta_C-\theta_K$ trajectory"),
              Line2D([], [], color=R.COLOR["none"], lw=2.6, label="ReLU steer, on the unsteered trajectory"),
              _handle(R.COLOR["difference_lora"], "s", r"LoRA, on the $\theta_C-\theta_K$ trajectory"),
              _handle(R.COLOR["none"], "s", "LoRA, on the unsteered trajectory"),
              Patch(facecolor=R.MUTED, alpha=0.35, edgecolor="none", label="band or bar: seed range")]
        _legend(sfc, hc, ncol=3)
        _finish(fig, [(axa[0], "a"), (axb[0], "b"), (axC, "c")], [(None, objs, 4), (None, glyphs, 3)])
        AC.save_figure(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig


# ---- figA_refusal_enrichment ---------------------------------------------------------------------------------------------------------
# The manuscript's canvas (appendix_geometry_style): 5.5 x 2.29 in, two legend rows on top (their top edges 0.00 and 0.20 in below the
# canvas's top, left edge 0.50 in), then two sub-axes 2.20 in wide and 1.45 in tall, their top edge 0.56 in down, 0.62 in from the left
# and 0.32 in apart. Within a component's x slot the objects sit at OBJ_DX and each seed's small mark ENRICH_SEED_DX right of its mean.
ENRICH_CANVAS_IN = (GS.TEXTWIDTH_IN, 2.29)
ENRICH_AXES_IN = dict(top=0.56, height=1.45, left=0.62, gap=0.32, width=2.20)
ENRICH_LEGEND_IN = dict(x=0.50, y=(0.0, 0.20))
ENRICH_PANELS = (("span", "Per-layer refusal span (9 positions)"), ("pc1", "First refusal PC (all layers, positions)"))
ENRICH_YLABEL = "Share of squared norm in\nrefusal subspace, ÷ random"
OBJ_DX = {"C": -0.24, "K": 0.0, "S": 0.24}
ENRICH_SEED_DX = 0.075


def refusal_enrichment_figure(sc: dict, name: str = FIG_ENRICHMENT, out_dirs=None, extra_dirs=()):
    """figA_refusal_enrichment as the manuscript prints it (appendix_geometry_style; the saved PNG is the manuscript's pixel for
    pixel): the enrichment over isotropic in each layer's refusal span (left) and on the first refusal PC (right), per component a grey
    random-direction band, each object's seed mean (large mark) and its seeds (small marks), one panel with no letter, the legend in
    two rows above the axes. Saved by GS.save into `out_dirs` (default R.FIGS) and the mirrors."""
    en = frame(sc, "refusal_subspace", "enrichment")
    band = enrichment_band(sc)
    W, H = ENRICH_CANVAS_IN
    ax_in, lg = ENRICH_AXES_IN, ENRICH_LEGEND_IN
    fx = lambda x: x / W
    fy = lambda y: 1 - y / H
    with GS.style():
        fig = plt.figure(figsize=(W, H))
        hs = [GS.object_handle("C", "relu", r"$\theta_C$ (constitution)"), GS.object_handle("K", "relu", r"$\theta_K$ (control)"),
              GS.object_handle("S", "relu", "safety positive control"),
              GS.shape_handle("relu", "ReLU steer"), GS.shape_handle("lora", "LoRA"),
              Line2D([], [], marker="o", ls="none", ms=GS.MS, mfc=GS.INK2, mec=GS.SURFACE, mew=0.5, label="seed mean"),
              Line2D([], [], marker="o", ls="none", ms=GS.MS_SEED, mfc=GS.INK2, mec=GS.SURFACE, mew=0.3, label="each seed"),
              Patch(facecolor=GS.BAND, edgecolor="none", label="random-direction 95 % range")]
        fig.legend(handles=hs[:5], loc="upper left", bbox_to_anchor=(fx(lg["x"]), fy(lg["y"][0])), ncol=5, frameon=False,
                   columnspacing=0.9, handletextpad=0.3, fontsize=GS.FS_SMALL)
        fig.legend(handles=hs[5:], loc="upper left", bbox_to_anchor=(fx(lg["x"]), fy(lg["y"][1])), ncol=3, frameon=False,
                   columnspacing=0.9, handletextpad=0.3, fontsize=GS.FS_SMALL)
        top, h, left, gap, w = ax_in["top"], ax_in["height"], ax_in["left"], ax_in["gap"], ax_in["width"]
        axS = fig.add_axes([fx(left), fy(top + h), fx(w), h / H])
        axP = fig.add_axes([fx(left + w + gap), fy(top + h), fx(w), h / H], sharey=axS)
        for ax, (which, lab) in zip((axS, axP), ENRICH_PANELS):
            for i, (fam, comp, _) in enumerate(COMPONENTS):
                b = band[band.family.eq(fam) & band.component.eq(comp) & band.which.eq(which)].iloc[0]
                ax.add_patch(plt.Rectangle((i - 0.40, b.lo), 0.80, b.hi - b.lo, facecolor=GS.BAND, edgecolor="none", zorder=0))
                for obj in ("C", "K", "S"):
                    g = en[en.family.eq(fam) & en.component.eq(comp) & en.which.eq(which) & en.obj.eq(obj)]
                    x = i + OBJ_DX[obj]
                    ax.plot([x], [g.enrichment.mean()], zorder=4, **GS.marker_kw(obj, fam))
                    if obj != "S":
                        for v in g.sort_values("seed").enrichment:
                            ax.plot([x + ENRICH_SEED_DX], [v], zorder=3, **GS.marker_kw(obj, fam, ms=GS.MS_SEED, seed=True))
            ax.axhline(1.0, color=GS.BASELINE, lw=0.7, zorder=1)
            ax.set_yscale("log")
            ax.set_xticks(range(len(COMPONENTS)))
            ax.set_xticklabels([c[2] for c in COMPONENTS], color=GS.INK, linespacing=1.0)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, len(COMPONENTS) - 0.5)
            GS.grid(ax, "y")
            GS.facet(ax, lab, y=1.06)
        axS.set_ylim(0.5, 12)
        axS.set_yticks([0.5, 1, 2, 4, 8])
        axS.set_yticklabels(["0.5", "1", "2", "4", "8"])
        axS.yaxis.set_minor_locator(plt.NullLocator())
        axS.set_ylabel(ENRICH_YLABEL, labelpad=2)
        plt.setp(axP.get_yticklabels(), visible=False)
        lo, hi = float(en.enrichment.min()), float(en.enrichment.max())
        if lo < 0.5 or hi > 12:
            raise GS.CanvasError(f"{name}: an enrichment of {lo:.3g}-{hi:.3g} leaves the fixed y range 0.5-12")
        GS.save(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig
