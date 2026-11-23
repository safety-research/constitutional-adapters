#!/usr/bin/env python
"""Build `appendix_geometry_forward.json`, the sidecar two judge-free geometry figures draw: figA_refusal_enrichment (weight geometry
against a multi-layer refusal geometry, Figure 9's lower half) and figA_displacement (forward passes on fixed text, notebook only).

WHAT IT HOLDS. Aggregates only, never activations: per seed and layer, the numbers each panel plots, plus the few levels and counts the
panels print. Every number is bound to the objects it measures by sha256 (the registry's `object_id`), and every object to the registry.
  displacement.kl          (a) mean per-token KL(p_X || p_unsteered), positions after the first, per family x text set x arm x seed
  displacement.kl_per_eval the same KL per evaluation inside each text set, and the first-reply-token KL (the text's per-evaluation claims)
  displacement.projection  (b) the change in the projection of the post-instruction state on the unit refusal direction of the same layer
                           (the block-output refit), per family x seed x prompt set x position (-5, -1) x layer x arm; the unsteered
                           mean projection of each prompt set at the same positions and layers; the refusal steer's per-prompt SE
  displacement.writes      (c) per layer, on the difference-object and unsteered trajectories: the ReLU gate correlation and the share
                           kappa_C of theta_C's realized write that theta_K's removes (ReLU at 32 layers, LoRA at its 8), seeds pooled
                           with the seed range; and the same pooled over layers
  refusal_subspace.*       (a) per-seed enrichment of the refusal-subspace share over isotropic, and each role's random-direction null;
                           (b) per-layer top principal-angle cosines of LoRA pairs, and the per-layer random-subspace null;
                           and the per-layer signed cosine of each ReLU write with the refusal direction at positions -5 and
                           -1, whose layer means the text quotes (not drawn)

WHERE THE NUMBERS COME FROM. The geometry analysis's two judge-free lanes (outside this repository), named relative to --source:
  gpu/geom-fwd-0074823/       the forward-pass job's aggregate (run geom-fwd-0074823, code 0074823c8): c2_kl.csv, c2_kl_per_eval.csv, c3_projection_summary.csv,
                              c1_{relu,lora}_{layers,pooled}.csv, sanity.json, refusal_directions_resid_post.pt (the block-output refit)
  gpu/runs/geom-fwd-0074823/shared/T1/shard_*.pt
                              the shared task's raw shards, read for two things the aggregate does not keep: the UNSTEERED mean
                              projection of each prompt set (c3_projection_summary.csv holds changes only) and the refusal steer's
                              per-prompt spread. The refusal steer's mean change is recomputed from them and must equal the aggregate's.
  gpu/inputs/texts_manifest.json   the text sets' sizes
  weights/b2_resid_pre_f5373ce7/   b2_summary.csv, b2_summary_by_role.csv, b2_control_vs_positive_control.csv, b2_relu_by_layer.csv,
                              b2_sign_by_layer_summary.csv, b2_checks.json (the refusal subspace: the paper fit's resid_pre means, sha f5373ce7)
  weights/b1/                 b1_pairs_by_layer.csv, b1_pairs_stacked.csv, b1_nulls.json, b1_checks.json (LoRA principal angles)
Every file is recorded with its sha256. Nothing is written unless: the job's own sanity checks all passed; every object id the lanes name is
in the registry under the expected paper label and fit checkpoint, with the derived objects' parents (-theta_K and the -0.5 refusal steer
are this seed's +1 object scaled by -1, the difference objects this seed's theta_C and theta_K); each pooled number the figures draw equals
the mean of its seeds (equal token counts) or the lane's own pooled table; and the recomputed refusal-steer changes equal the aggregate's.

usage, from the repository root (torch reads the shards, so a venv synced with `--extra cpu`; the registry is opened read-only):
  PYTHONPATH=$PWD/src .venv/bin/python paper/notebooks/data/build_appendix_geometry_forward.py --source DIR [--registry DB] [--check]
--source is the lane's analysis directory, holding gpu/ and weights/ (default $APPENDIX_GEOMETRY_FORWARD_SOURCE); --registry defaults to
$REGISTRY_ROOT/registry.sqlite (REGISTRY_ROOT default ~/registry); --check rebuilds in memory and compares with the committed file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "appendix_geometry_forward.json"
SCHEMA = "appendix_geometry_forward/1"
SCRIPT = "paper/notebooks/data/build_appendix_geometry_forward.py"
RUN = "geom-fwd-0074823"
GPU = f"gpu/{RUN}"
SHARDS = f"gpu/runs/{RUN}/shared/T1/shard_*.pt"
B2 = "weights/b2_resid_pre_f5373ce7"
B1 = "weights/b1"
SOURCES = {
    "kl": f"{GPU}/c2_kl.csv", "kl_per_eval": f"{GPU}/c2_kl_per_eval.csv", "projection": f"{GPU}/c3_projection_summary.csv",
    "relu_layers": f"{GPU}/c1_relu_layers.csv", "lora_layers": f"{GPU}/c1_lora_layers.csv",
    "relu_pooled": f"{GPU}/c1_relu_pooled.csv", "lora_pooled": f"{GPU}/c1_lora_pooled.csv",
    "sanity": f"{GPU}/sanity.json", "directions_c4": f"{GPU}/refusal_directions_resid_post.pt",
    "shared_meta": f"gpu/runs/{RUN}/shared/task_meta.json", "texts": "gpu/inputs/texts_manifest.json",
    "b2_summary": f"{B2}/b2_summary.csv", "b2_by_role": f"{B2}/b2_summary_by_role.csv",
    "b2_cvp": f"{B2}/b2_control_vs_positive_control.csv", "b2_relu_by_layer": f"{B2}/b2_relu_by_layer.csv",
    "b2_sign": f"{B2}/b2_sign_by_layer_summary.csv", "b2_checks": f"{B2}/b2_checks.json",
    "b1_pairs": f"{B1}/b1_pairs_by_layer.csv", "b1_stacked": f"{B1}/b1_pairs_stacked.csv",
    "b1_nulls": f"{B1}/b1_nulls.json", "b1_checks": f"{B1}/b1_checks.json",
}
JUDGE = {"model": "none",
         "note": "judge-free: teacher-forced forward passes on fixed text (the unsteered model's replies) and weight geometry; "
                 "no generation is graded, so no judge enters any number here"}
ABOUT = ("Qwen3.5-4B (revision 851bf6e8), the paper's base-to-instruct ReLU steers and LoRAs, seeds 1-3: theta_C (constitution), theta_K "
         "(control), -theta_K (theta_K with its write negated, applied as stored) and theta_C - theta_K (the difference object), plus the "
         "refusal steer (Arditi et al., 2024) at c = +0.5 and -0.5 and the safety positive controls (one ReLU steer and one LoRA distilled "
         "from a safety-refusal system prompt, fit on the instruct checkpoint). Aggregates of the geometry analysis's forward-pass job "
         f"({RUN}) and weight analyses, bound to the objects by sha256.")
SEEDS = ("1", "2", "3")
FAMILIES = ("relu", "lora")
ARMS = ("pK", "mK", "C", "V")
STEER_ARMS = ("pA", "mA")
EVALS = ("arditi_harmful_val", "arditi_harmless_val")
POSITIONS = (-5, -1)
N_POST = 9                                    # post-instruction template positions -9..-1
N_LAYERS = 32
D_MODEL = 2560
ROLE_LABEL = {("relu", "C"): "constitution_relu", ("relu", "K"): "control_relu", ("relu", "mK"): "control_relu",
              ("relu", "V"): "difference_relu", ("lora", "C"): "constitution_lora", ("lora", "K"): "control_lora",
              ("lora", "mK"): "control_lora", ("lora", "V"): "difference_lora", ("arditi", "pA"): "arditi_allpos",
              ("arditi", "mA"): "arditi_allpos", ("relu", "S"): "safety_relu", ("lora", "S"): "safety_lora"}
FIT_MODEL = {"C": "qwen35_4b_base", "K": "qwen35_4b_base", "mK": "qwen35_4b_base", "V": "qwen35_4b_base",
             "pA": "qwen35_4b_it", "mA": "qwen35_4b_it", "S": "qwen35_4b_it"}
ENRICH_COMPONENTS = (("relu", "read", "frac_span", "frac_top1"), ("relu", "write", "frac_span", "frac_top1"),
                     ("lora", "write_post", "dwfrac_span", "dwfrac_top1"))
PAIR_TYPES = ("C-K same seed", "C-K cross seed", "C-C seed ceiling", "K-K seed ceiling", "C-safety (other lane)", "K-safety (other lane)")
REL = 1e-9


class Refused(SystemExit):
    pass


def refuse(msg: str):
    raise Refused(f"REFUSED: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def close(a: float, b: float, tol: float = REL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def block(columns, rows) -> dict:
    return {"columns": list(columns), "rows": [list(r) for r in rows]}


# ---- the sources -------------------------------------------------------------------------------------------------------------------
def load_sources(src: Path) -> tuple[dict, list]:
    miss = [rel for rel in SOURCES.values() if not (src / rel).is_file()]
    shards = sorted(glob.glob(str(src / SHARDS)))
    if not shards:
        miss.append(SHARDS)
    if miss:
        refuse(f"--source {src} lacks the lane's file(s) {miss}")
    d = {}
    for k, rel in SOURCES.items():
        p = src / rel
        if p.suffix == ".json":
            d[k] = json.loads(p.read_text())
        elif p.suffix == ".csv":
            d[k] = pd.read_csv(p, dtype={"seed": str, "layer": str} if k in ("relu_layers", "lora_layers", "relu_pooled", "lora_pooled")
                               else {"seed": str}, low_memory=False)
    files = [dict(name=rel, sha256=sha256(src / rel)) for rel in SOURCES.values()]
    files += [dict(name=os.path.relpath(p, src), sha256=sha256(Path(p))) for p in shards]
    d["_shards"] = shards
    return d, files


# ---- objects, bound to the registry ------------------------------------------------------------------------------------------------
def lane_objects(L: dict) -> list:
    """Every object the two lanes measured: role, family, seed and sha256, from the job's sanity record and the weight lanes' checks."""
    san = L["sanity"]
    if not all(san["checks"].values()):
        refuse(f"{SOURCES['sanity']}: the job's sanity checks did not all pass: {[k for k, v in san['checks'].items() if not v]}")
    objs = {}

    def add(oid, role, family, seed, used):
        key = oid
        if key in objs:
            o = objs[key]
            if (o["role"], o["family"], o["seed"]) != (role, family, seed):
                refuse(f"object {oid} named twice with different roles: {o} vs {(role, family, seed)}")
            if used not in o["used_by"]:
                o["used_by"].append(used)
        else:
            objs[key] = dict(object_id=oid, role=role, family=family, seed=seed, used_by=[used])

    for task, roles in san["details"]["objects"].items():
        if task == "shared":
            for r, oid in roles.items():
                add(oid, r, "arditi", "", "displacement")
        else:
            fam, s = task.split("_s")
            for r, oid in roles.items():
                add(oid, r, fam, s, "displacement")
    for chk in ("b2_checks", "b1_checks"):
        for key, o in L[chk]["objects"].items():
            name, seed = key.split(":")
            role, fam = name.split("_")
            role = {"Km1": "mK"}.get(role, role)
            add(o["sha256"], role, fam, "" if seed == "-" else seed.lstrip("s"), "refusal_subspace")
    return sorted(objs.values(), key=lambda o: (o["family"], o["role"], o["seed"]))


def bind_registry(objs: list, db: Path) -> int:
    if not db.is_file():
        refuse(f"registry {db} not found; the sidecar binds every object to it (pass --registry)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    by_role = {(o["family"], o["role"], o["seed"]): o["object_id"] for o in objs}
    bad = []
    for o in objs:
        r = con.execute("SELECT paper_label, fit_model_id, fit_replicate_label, derivation, parent_object_ids_json, base_scale, status "
                        "FROM object WHERE object_id = ?", (o["object_id"],)).fetchone()
        if r is None:
            bad.append(f"{o['object_id']} ({o['family']} {o['role']} s{o['seed']}): not in the registry")
            continue
        label, fit_model, rep, deriv, parents_json, base_scale, status = r
        parents = json.loads(parents_json) if parents_json else []
        want_label = ROLE_LABEL[(o["family"], o["role"])]
        if label != want_label or fit_model != FIT_MODEL[o["role"]]:
            bad.append(f"{o['object_id']}: registry says {label} / {fit_model}, the lane's role says {want_label} / {FIT_MODEL[o['role']]}")
        if o["role"] in ("C", "K") and rep != f"s{o['seed']}":
            bad.append(f"{o['object_id']}: registry replicate {rep!r}, the lane's seed s{o['seed']}")
        if o["role"] == "mK":
            want = [by_role.get((o["family"], "K", o["seed"]))]
            if deriv != "dose_scale" or parents != want or base_scale != -1.0:
                bad.append(f"{o['object_id']}: -theta_K should be dose_scale -1 of {want}, registry {deriv} {base_scale} {parents}")
        if o["role"] == "V":
            want = [by_role.get((o["family"], "C", o["seed"])), by_role.get((o["family"], "K", o["seed"]))]
            if parents != want:
                bad.append(f"{o['object_id']}: the difference object's parents should be {want}, registry {parents}")
        if o["role"] == "mA":
            want = [by_role.get(("arditi", "pA", ""))]
            if deriv != "dose_scale" or parents != want or base_scale != -1.0:
                bad.append(f"{o['object_id']}: the -0.5 refusal steer should be dose_scale -1 of {want}, registry {deriv} {base_scale} {parents}")
        o["registry"] = dict(paper_label=label, fit_model_id=fit_model, fit_replicate_label=rep, derivation=deriv, parents=parents,
                             base_scale=base_scale, status=status)
    if bad:
        refuse("objects do not bind to the registry: " + "; ".join(bad))
    head = int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
    con.close()
    return head


# ---- displacement ----------------------------------------------------------------------------------------------------------------
def kl_block(L: dict) -> dict:
    d = L["kl"]
    d = d[d.classset.eq("nonpos0")]
    out = []
    for fam in FAMILIES:
        for st in ("T1", "T2", "T3"):
            for arm in ARMS:
                g = d[d.family.eq(fam) & d.set.eq(st) & d.cond.eq(arm)].set_index("seed")
                missing = [s for s in SEEDS + ("pooled",) if s not in g.index]
                if missing:
                    refuse(f"{SOURCES['kl']}: {fam} {st} {arm} lacks seeds {missing}")
                seed_mean = float(np.mean([g.at[s, "mean_kl"] for s in SEEDS]))
                if not close(seed_mean, float(g.at["pooled", "mean_kl"]), 1e-9) or g.loc[list(SEEDS), "n"].nunique() != 1:
                    refuse(f"{SOURCES['kl']}: {fam} {st} {arm}: the pooled KL {g.at['pooled', 'mean_kl']} is not the mean of equal-n seeds {seed_mean}")
                for s in SEEDS + ("pooled",):
                    out.append([fam, st, arm, s, float(g.at[s, "mean_kl"]), int(g.at[s, "n"])])
    a = d[d.family.eq("arditi") & d.set.eq("T1") & d.cond.isin(STEER_ARMS)]
    for arm in STEER_ARMS:
        r = a[a.cond.eq(arm)]
        if len(r) != 1:
            refuse(f"{SOURCES['kl']}: arditi T1 {arm}: {len(r)} rows")
        out.append(["arditi", "T1", arm, "", float(r.mean_kl.iat[0]), int(r.n.iat[0])])
    return block(["family", "set", "arm", "seed", "mean_kl", "n_tokens"], out)


def kl_per_eval_block(L: dict) -> dict:
    """Per evaluation inside each text set: the KL over tokens after the first ((sum_kl_reply + sum_kl_prompt) / (n_kl_reply +
    n_kl_prompt), the aggregate's headline rule) and the first-reply-token KL, per family x seed x arm. Token-weighted over a set's
    evaluations they must give the set's KL in c2_kl.csv."""
    pe = L["kl_per_eval"]
    pe = pe[pe.family.isin(FAMILIES) & pe.set.isin(["T2", "T3"]) & pe.cond.isin(ARMS) & pe.seed.isin(SEEDS)].copy()
    tot = pe.groupby(["family", "seed", "set", "cond"])[["sum_kl_reply", "sum_kl_prompt", "n_kl_reply", "n_kl_prompt"]].sum()
    kl = L["kl"]
    kl = kl[kl.classset.eq("nonpos0")].set_index(["family", "seed", "set", "cond"]).mean_kl
    for key, r in tot.iterrows():
        m = (r.sum_kl_reply + r.sum_kl_prompt) / (r.n_kl_reply + r.n_kl_prompt)
        if not close(m, float(kl[key])):
            refuse(f"{SOURCES['kl_per_eval']}: {key}: the token-weighted KL over evaluations {m} is not c2_kl.csv's {kl[key]}")
    if len(tot) != len(FAMILIES) * len(SEEDS) * 2 * len(ARMS):
        refuse(f"{SOURCES['kl_per_eval']}: {len(tot)} family x seed x set x arm groups")
    rows = []
    for r in pe.sort_values(["family", "seed", "set", "eval", "cond"]).itertuples():
        n = r.n_kl_reply + r.n_kl_prompt
        rows.append([r.family, r.seed, r.set, r.eval, r.cond, float((r.sum_kl_reply + r.sum_kl_prompt) / n), float(r.first_token_kl),
                     int(r.n_items), int(n)])
    return block(["family", "seed", "set", "eval", "arm", "kl", "first_token_kl", "n_items", "n_tokens"], rows)


def shard_projections(L: dict, c4: dict) -> dict:
    """The shared task's T1 shards projected on the block-output refit (the aggregate's own recipe: the raw harmful-minus-harmless means
    normalised in float64, cast to float32; bf16 states cast to float32), per condition: (uids, [n, 9, 32])."""
    import torch
    md = c4["mean_diffs"].double()
    dn = (md / md.norm(dim=-1, keepdim=True).clamp_min(1e-12)).float()
    out = {c: ([], []) for c in ("clean",) + STEER_ARMS}
    for f in L["_shards"]:
        s = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        it = s["items"]
        for c in out:
            out[c][0].extend(it[f"uids/{c}"])
            out[c][1].append(torch.einsum("npld,pld->npl", it[f"states/{c}"].float(), dn).numpy())
    return {c: (u, np.concatenate(a)) for c, (u, a) in out.items()}


def projection_block(L: dict, src: Path) -> dict:
    import torch
    c4 = torch.load(src / SOURCES["directions_c4"], map_location="cpu", weights_only=False)
    if tuple(c4["mean_diffs"].shape) != (N_POST, N_LAYERS, D_MODEL):
        refuse(f"{SOURCES['directions_c4']}: mean_diffs shape {tuple(c4['mean_diffs'].shape)}")
    proj = shard_projections(L, c4)
    cu, ca = proj["clean"]
    ev = np.array([u.split("/")[1] for u in cu])
    n_by_eval = {e: int((ev == e).sum()) for e in EVALS}
    if any(v == 0 for v in n_by_eval.values()) or len(ev) != sum(n_by_eval.values()):
        refuse(f"{SHARDS}: unexpected T1 prompt sets {pd.Series(ev).value_counts().to_dict()}")
    unsteered = []
    for e in EVALS:
        m = ca[ev == e].mean(0)                                    # [9, 32]
        for pos in POSITIONS:
            for layer in range(N_LAYERS):
                unsteered.append([e, pos, layer, float(m[pos + N_POST, layer])])
    # the aggregate's changes
    d = L["projection"]
    d = d[d.directions.eq("c4_resid_post") & d.set.eq("T1") & d["eval"].isin(EVALS) & d.position.isin(POSITIONS)]
    rows = []
    for fam in FAMILIES:
        for s in SEEDS:
            g = d[d.family.eq(fam) & d.seed.eq(s)]
            for e in EVALS:
                for pos in POSITIONS:
                    q = g[g["eval"].eq(e) & g.position.eq(pos)].set_index("layer").sort_index()
                    if list(q.index) != list(range(N_LAYERS)):
                        refuse(f"{SOURCES['projection']}: {fam} s{s} {e} position {pos}: layers {list(q.index)}")
                    for layer, r in q.iterrows():
                        vals = [float(r[a]) for a in ARMS]
                        if not (close(r.odd_K, (r.pK - r.mK) / 2, 1e-6) and close(r.even_K, (r.pK + r.mK) / 2, 1e-6)):
                            refuse(f"{SOURCES['projection']}: {fam} s{s} {e} {pos} L{layer}: even/odd do not decompose pK, mK")
                        rows.append([fam, s, e, pos, int(layer)] + vals)
    # the refusal steer: the aggregate's mean change, recomputed from the shards, and its per-prompt SE
    steer = []
    a = d[d.family.eq("arditi")]
    cidx = {u: i for i, u in enumerate(cu)}
    for e in EVALS:
        for pos in POSITIONS:
            q = a[a["eval"].eq(e) & a.position.eq(pos)].set_index("layer").sort_index()
            if list(q.index) != list(range(N_LAYERS)):
                refuse(f"{SOURCES['projection']}: arditi {e} position {pos}: layers {list(q.index)}")
            for arm in STEER_ARMS:
                uu, arr = proj[arm]
                idx = [cidx[u] for u in uu]
                dif = arr - ca[idx]
                evs = np.array([u.split("/")[1] for u in uu])
                x = dif[evs == e][:, pos + N_POST, :]                  # [n, 32]
                n = x.shape[0]
                mean, se = x.mean(0), x.std(0, ddof=1) / math.sqrt(n)
                for layer in range(N_LAYERS):
                    agg = float(q.at[layer, arm])
                    if abs(mean[layer] - agg) > 1e-4 * max(1.0, abs(agg)):
                        refuse(f"refusal steer {arm} {e} {pos} L{layer}: the shards give {mean[layer]:.6f}, the aggregate {agg:.6f}")
                    steer.append([e, pos, layer, arm, agg, float(se[layer]), n])
    return {"directions": "c4_resid_post",
            "directions_note": ("the refusal directions refit at the block OUTPUT (resid_post, index l = the output of decoder block l) "
                                "from the paper fit's train prompts with its filter, unit-normalised; the fit's own resid_pre means "
                                "shifted by one block agree to cosine 0.99999"),
            "directions_sha256": sha256(src / SOURCES["directions_c4"]),
            "texts": {e: n_by_eval[e] for e in EVALS},
            "quantity": ("mean over prompts of the projection of the post-instruction residual state on the unit refusal direction of "
                         "the same layer and position; a change is (steered - unsteered) per prompt, averaged"),
            "unsteered": block(["eval", "position", "layer", "mean_proj"], unsteered),
            "delta": block(["family", "seed", "eval", "position", "layer"] + list(ARMS), rows),
            "refusal_steer": block(["eval", "position", "layer", "arm", "delta", "se", "n_prompts"], steer),
            "refusal_steer_layer": int(L["shared_meta"]["relations"]["layers"][0])}


def writes_block(L: dict) -> dict:
    def sel(d):
        return d[d.seed.eq("pooled") & d.set.eq("T2+T3") & d.classset.eq("nonpos0") & d.traj.isin(["V", "clean"])]
    rl, ll = sel(L["relu_layers"]), sel(L["lora_layers"])
    relu = []
    for traj in ("V", "clean"):
        g = rl[rl.traj.eq(traj)].assign(layer=lambda f: f.layer.astype(int)).sort_values("layer")
        if list(g.layer) != list(range(N_LAYERS)) or not (g.n_seeds.astype(float) == 3).all():
            refuse(f"{SOURCES['relu_layers']}: {traj}: layers {list(g.layer)} / seeds {g.n_seeds.unique()}")
        for r in g.itertuples():
            relu.append([traj, int(r.layer), float(r.corr_g), float(r.corr_g_seed_min), float(r.corr_g_seed_max),
                         float(r.kappa_C), float(r.kappa_C_seed_min), float(r.kappa_C_seed_max)])
    lora = []
    for traj in ("V", "clean"):
        g = ll[ll.traj.eq(traj)].assign(layer=lambda f: f.layer.astype(int)).sort_values("layer")
        if list(g.layer) != [3, 7, 11, 15, 19, 23, 27, 31] or not (g.n_seeds.astype(float) == 3).all():
            refuse(f"{SOURCES['lora_layers']}: {traj}: layers {list(g.layer)}")
        for r in g.itertuples():
            lora.append([traj, int(r.layer), float(r.kappa_C), float(r.kappa_C_seed_min), float(r.kappa_C_seed_max)])
    pooled = []
    for fam, key, cols in (("relu", "relu_pooled", ("corr_g", "kappa_C")), ("lora", "lora_pooled", ("kappa_C",))):
        p = L[key]
        p = p[p.seed.eq("pooled") & p.set.eq("T2+T3") & p.classset.eq("nonpos0") & p.layer.eq("pooled")].set_index("traj")
        for traj in ("V", "clean"):
            for c in cols:
                pooled.append([fam, traj, c, float(p.at[traj, c])])
    return {"texts": "T2+T3 (attack and benign prompts with the unsteered model's replies), tokens after the first",
            "relu": block(["traj", "layer", "corr_g", "corr_g_seed_min", "corr_g_seed_max", "kappa_C", "kappa_C_seed_min",
                           "kappa_C_seed_max"], relu),
            "lora": block(["traj", "layer", "kappa_C", "kappa_C_seed_min", "kappa_C_seed_max"], lora),
            "pooled_over_layers": block(["family", "traj", "stat", "value"], pooled)}


# ---- refusal subspace ------------------------------------------------------------------------------------------------------------
def subspace_block(L: dict) -> dict:
    sm, nb, cvp = L["b2_summary"], L["b2_by_role"], L["b2_cvp"].set_index(["family", "component"])
    enrich, null = [], []
    for fam, comp, m_span, m_top1 in ENRICH_COMPONENTS:
        for metric, which in ((m_span, "span"), (m_top1, "pc1")):
            for obj in ("C", "K", "S"):
                role = f"{obj}_{fam}"
                g = sm[sm.family.eq(fam) & sm.component.eq(comp) & sm.metric.eq(metric) & sm.role.eq(role)].sort_values("seed")
                want = ["-"] if obj == "S" else ["s1", "s2", "s3"]
                if list(g.seed) != want:
                    refuse(f"{SOURCES['b2_summary']}: {fam} {comp} {metric} {role}: seeds {list(g.seed)}")
                ref = float(cvp.at[(fam, comp), f"{obj}_{which}_enrich_mean"])
                if not close(float(g.enrichment.mean()), ref):
                    refuse(f"{SOURCES['b2_summary']}: {role} {comp} {which}: seed mean {g.enrichment.mean()} vs the lane's {ref}")
                for r in g.itertuples():
                    enrich.append([fam, comp, which, obj, "" if r.seed == "-" else r.seed.lstrip("s"), float(r.enrichment)])
                q = nb[nb.family.eq(fam) & nb.component.eq(comp) & nb.metric.eq(metric) & nb.role.eq(role)]
                if len(q) != 1:
                    refuse(f"{SOURCES['b2_by_role']}: {fam} {comp} {metric} {role}: {len(q)} rows")
                q = q.iloc[0]
                null.append([fam, comp, which, obj, float(q.expected_isotropic), float(q["null_q0.025"]), float(q["null_q0.975"]),
                             float(q.p_null_max), int(q.n_seeds)])
    rl = L["b2_relu_by_layer"]
    cos = []
    for role, obj in (("K_relu", "K"), ("C_relu", "C"), ("S_relu", "S")):
        g = rl[rl.role.eq(role) & rl.component.eq("write")].dropna(subset=["cos_pos-5", "cos_pos-1"]).sort_values(["seed", "layer"])
        want = ["-"] if obj == "S" else ["s1", "s2", "s3"]
        if sorted(g.seed.unique()) != want or (g.groupby("seed").size() != N_LAYERS - 1).any():
            refuse(f"{SOURCES['b2_relu_by_layer']}: {role} write: seeds {sorted(g.seed.unique())}, layers per seed {g.groupby('seed').size().to_dict()}")
        for _, r in g.iterrows():
            cos.append([obj, "" if r.seed == "-" else r.seed.lstrip("s"), int(r.layer), float(r["cos_pos-5"]), float(r["cos_pos-1"])])
    # the lane's per-layer sign summary must be what these rows give
    sg = L["b2_sign"].set_index(["role", "component", "pos"])
    cf = pd.DataFrame(cos, columns=["obj", "seed", "layer", "cos_pos-5", "cos_pos-1"])
    for obj in ("K", "C"):
        for pos in (-5, -1):
            lm = cf[cf.obj.eq(obj)].groupby("layer")[f"cos_pos{pos}"].mean().mean()
            ref = float(sg.loc[(f"{obj}_relu", "write", pos), "layer_mean_of_seed_mean"])
            if not close(lm, ref):
                refuse(f"{SOURCES['b2_sign']}: {obj} pos {pos}: layer mean of seed means {lm} vs the lane's {ref}")
    pl = L["b1_pairs"]
    pairs = []
    for r in pl[pl.pair_type.isin(PAIR_TYPES)].sort_values(["pair_type", "x", "y", "layer"]).itertuples():
        pairs.append([r.pair_type, r.x, r.y, int(r.layer), float(r.out_top), float(r.in_top), float(r.cos_F)])
    if len(pairs) != len(pl):
        refuse(f"{SOURCES['b1_pairs']}: pair types outside {PAIR_TYPES}: {sorted(set(pl.pair_type) - set(PAIR_TYPES))}")
    st = L["b1_stacked"]
    stacked = [[r.pair_type, r.x, r.y, float(r.cos_F)] for r in st.sort_values(["pair_type", "x", "y"]).itertuples()]
    nl = L["b1_nulls"]
    haar = {k: {q: float(nl[k]["top"][q]) for q in ("q0.025", "q0.5", "q0.975")} | {"n": int(nl[k]["top"]["n"])}
            for k in ("out_2560", "in_4096")}
    dj = L["b2_checks"]["directions"]
    return {"directions": {"site": dj["site"], "sha256": dj["sha256"], "positions": dj["positions"],
                           "n_vectors": int(dj["global_pca_n_vectors"]), "pc1_explained": float(dj["global_pca_explained"]["top1"]),
                           "note": ("the paper fit's harmful-minus-harmless means at every layer but the last and the 9 post-instruction "
                                    "positions (resid_pre); a ReLU write at block l pairs with the direction at index l + 1")},
            "d_model": D_MODEL,
            "enrichment": block(["family", "component", "which", "obj", "seed", "enrichment"], enrich),
            "enrichment_null": block(["family", "component", "which", "obj", "expected_isotropic", "null_q0.025", "null_q0.975",
                                      "p_null_max", "n_seeds"], null),
            "signed_cosine": block(["obj", "seed", "layer", "cos_pos-5", "cos_pos-1"], cos),
            "principal_angles": block(["pair_type", "x", "y", "layer", "out_top", "in_top", "cos_F"], pairs),
            "frobenius_stacked": block(["pair_type", "x", "y", "cos_F"], stacked),
            "haar_null": haar}


# ---- assemble, write, check ------------------------------------------------------------------------------------------------------
def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def build(src: Path, db: Path) -> dict:
    L, files = load_sources(src)
    objs = lane_objects(L)
    head = bind_registry(objs, db)
    texts = L["texts"]
    n_texts = {"T2": int(sum(v["chosen"] for v in texts["t2_selection_stats"].values())),
               "T3": int(sum(v["chosen"] for v in texts["t3_selection_stats"].values()))}
    san = L["sanity"]
    meta = L["shared_meta"]
    out = {
        "schema": SCHEMA,
        "about": ABOUT,
        "judge": JUDGE,
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": SCRIPT, "script_sha256": sha256(Path(__file__)),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", SCRIPT)),
            "lanes": ("the geometry analysis's judge-free lanes: the forward-pass job and the weight analyses; their "
                      "files are outside this repository and are named here relative to the lanes' analysis directory, each with its sha256"),
            "gpu_run": {"name": RUN, "code_sha": meta["env"]["code_sha"], "code_version": meta["env"]["code_version"],
                        "texts_sha256": meta["c4"]["texts_sha256"], "model": "Qwen/Qwen3.5-4B",
                        "sanity_checks_passed": sorted(k for k, v in san["checks"].items() if v)},
            "weights": {"refusal_directions_sha256": L["b2_checks"]["directions"]["sha256"], "b2_seed": L["b2_checks"].get("seed"),
                        "b2_n_null": L["b2_checks"].get("n_null"), "b1_nulls_seed": L["b1_nulls"].get("seed")},
            "registry": {"opened": "read-only", "head_at_build": head},
            "source_files": files,
        },
        "objects": objs,
        "texts": n_texts,
        "displacement": {"kl": kl_block(L), "kl_per_eval": kl_per_eval_block(L), "projection": projection_block(L, src),
                         "writes": writes_block(L)},
        "refusal_subspace": subspace_block(L),
    }
    return out


def _dump(obj, ind: int = 0) -> str:
    """JSON with one table row per line (a `rows` list of lists, or a list of flat dicts), everything else indented."""
    pad, pad1 = " " * ind, " " * (ind + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        if all(not isinstance(v, (dict, list)) for v in obj.values()) and len(json.dumps(obj)) < 160:
            return json.dumps(obj, ensure_ascii=False)
        items = [f"{pad1}{json.dumps(k)}: {_dump(v, ind + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj, ensure_ascii=False)
        return "[\n" + ",\n".join(pad1 + _dump(v, ind + 1) for v in obj) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


VOLATILE = ("generated_utc", "git_commit", "git_dirty", "script_sha256", "head_at_build")


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in VOLATILE}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=os.environ.get("APPENDIX_GEOMETRY_FORWARD_SOURCE"))
    ap.add_argument("--registry", default=os.path.join(os.path.expanduser(os.environ.get("REGISTRY_ROOT", "~/registry")), "registry.sqlite"))
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the committed file")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if not a.source:
        refuse("no --source (the lanes' analysis directory, holding gpu/ and weights/) and APPENDIX_GEOMETRY_FORWARD_SOURCE is unset")
    out = build(Path(os.path.expanduser(a.source)), Path(os.path.expanduser(a.registry)))
    text = _dump(out) + "\n"
    json.loads(text)                                            # the writer's output parses back
    if a.check:
        old = json.loads(Path(a.out).read_text())
        if _strip(old) != _strip(json.loads(text)):
            diff = [k for k in set(_strip(old)) | set(_strip(out)) if _strip(old).get(k) != _strip(json.loads(text)).get(k)]
            print(f"DIFFERS from {a.out}: top-level keys {sorted(diff)}")
            return 1
        print(f"{a.out}: matches a fresh build ({len(out['objects'])} objects, {len(out['provenance']['source_files'])} source files)")
        return 0
    Path(a.out).write_text(text)
    pr = out["displacement"]["projection"]
    print(f"wrote {a.out} ({len(text) / 1e3:.0f} kB): {len(out['objects'])} objects bound to the registry (head "
          f"{out['provenance']['registry']['head_at_build']}), {len(out['provenance']['source_files'])} source files; projection rows "
          f"{len(pr['delta']['rows'])} + refusal steer {len(pr['refusal_steer']['rows'])}; KL rows {len(out['displacement']['kl']['rows'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
