#!/usr/bin/env python
"""Build `appendix_geometry_overlap.json`, the sidecar the overlap figure of the control-directions appendix draws (figA_cK_overlap):
how much the constitution object theta_C and the control object theta_K overlap, layer by layer, against seed ceilings, an unrelated
safety-refusal object and random.

WHAT IT HOLDS. Per-pair, per-layer values only, never weights:
  lora_overlap   (a) per LoRA layer (3, 7, ..., 31) and space, the share of one LoRA's update ||dW_l||_F^2 that lies inside another
                 LoRA's rank-4 output space (span of B_l, "write") or input space (span of A_l, "read"), for every pair the figure draws
                 a mean and range over: theta_C in theta_K and theta_K in theta_C (9 seed pairs each), a seed in another seed of the
                 same object (6 ordered pairs per object), theta_C or theta_K in the safety LoRA (3 each), and the reverse, the safety
                 LoRA in theta_C or theta_K (3 each, not drawn)
  lora_null      per measured LoRA, layer and space, the quantiles of the same share when the other subspace is a Haar-random
                 4-dimensional one (2,000 draws per object, seed 1), from which the figure's grey band is the envelope over the six
                 theta_C and theta_K seeds
  relu_cosine    (b) per layer (0-31) and vector, the cosine between two ReLU steers' read (gate) vectors and write vectors: theta_C
                 against theta_K (9 seed pairs), seed pairs of one object (3 each), the safety ReLU steer against each object's 3 seeds
Every object is pinned by sha256 (the registry's object_id) and bound to the registry: the LoRAs by the null file, which names each
measured object's sha256, the ReLU steers by the committed layer-12 cosine data `relu_arditi_cosines.json`, which names theirs.

WHERE THE NUMBERS COME FROM. The geometry analysis's figure build (outside this repository) writes three CSVs beside the figure, named
relative to --source and recorded with their sha256:
  figA_cK_overlap_pairs.csv    every pair value per layer: panel (a) read from the weight lane's b1_pairs_by_layer.csv (the
                               one-sided overlap columns ov_out_only_* and ov_in_only_*, which the build recomputed from the object files),
                               panel (b) computed from the object files (torch cosine_similarity in float32)
  figA_cK_overlap_null.csv     the per-object random-subspace quantiles (the weight lane's pair_null)
  figA_cK_overlap_values.csv   every drawn mean, min and max and the bands; read only to check that the figure this sidecar draws is
                               the figure the build drew (every value within the two CSVs' printed precision of 8 significant digits)
Nothing is written unless: every pair, layer, space and series is present with the figure's counts; every share lies in [0, 1] and
every cosine in [-1, 1]; the layer-12 cosines equal those of `relu_arditi_cosines.json` to the CSV's 8 significant digits, object for
object (the build itself reproduced them exactly from the object files); every object id is
in the registry under the expected paper label, fit checkpoint and seed; and the drawn values recompute from the pairs and the null.

usage, from the repository root (no torch; the registry is opened read-only):
  PYTHONPATH=$PWD/src .venv/bin/python paper/notebooks/data/build_appendix_geometry_overlap.py --source DIR [--registry DB] [--check]
--source is the figure build's output directory, holding the three CSVs (default $APPENDIX_GEOMETRY_OVERLAP_SOURCE); --registry defaults
to $REGISTRY_ROOT/registry.sqlite (REGISTRY_ROOT default ~/registry); --check rebuilds in memory and compares with the committed file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_appendix_geometry_forward as BF  # noqa: E402  (its file, registry and JSON helpers)

REPO = HERE.parents[2]
OUT = HERE / "appendix_geometry_overlap.json"
COSINES = HERE / "relu_arditi_cosines.json"
SCHEMA = "appendix_geometry_overlap/1"
SCRIPT = "paper/notebooks/data/build_appendix_geometry_overlap.py"
FILES = {"pairs": "figA_cK_overlap_pairs.csv", "null": "figA_cK_overlap_null.csv", "values": "figA_cK_overlap_values.csv"}
JUDGE = {"model": "none",
         "note": "judge-free: weight geometry of the fitted objects; no generation is graded, so no judge enters any number here"}
ABOUT = ("Qwen3.5-4B, the paper's base-to-instruct ReLU steers and LoRAs, seeds 1-3: theta_C (constitution) and theta_K (control), with "
         "a safety-refusal ReLU steer and LoRA (distilled from a safety-refusal system prompt on the instruct checkpoint) as the "
         "unrelated reference. Per-pair, per-layer overlaps of the LoRA updates' output and input spaces and cosines of the ReLU "
         "steers' read and write vectors, with the LoRA overlaps' random-subspace null; bound to the objects by sha256.")
SEEDS = ("1", "2", "3")
D_MODEL, D_IN = 2560, 4096                     # the residual width (LoRA output, ReLU vectors) and o_proj's input width (LoRA input)
RANK = 4
LORA_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
N_LAYERS = 32
N_NULL, NULL_SEED = 2000, 1
# series -> pairs per (space, layer)
LORA_SERIES = {"CinK": 9, "KinC": 9, "CC": 6, "KK": 6, "CinS": 3, "KinS": 3, "SinC": 3, "SinK": 3}
RELU_SERIES = {"CK": 9, "CC": 3, "KK": 3, "SC": 3, "SK": 3}
COS_LABEL = {"constitution_relu": "C", "control_relu": "K", "safety_relu": "S"}
REL, ABS = 1e-6, 1e-8                           # the values and pairs CSVs are printed at 8 significant digits
LAYER12_REL = 1e-7                              # so is the pairs CSV: the layer-12 cosines agree with the cosine data to that precision


def key(role: str, family: str, seed: str) -> str:
    """The figure build's object key: role, family and seed, as `C_lora:s1` or `S_relu:-`."""
    return f"{role}_{family}:{('s' + seed) if seed else '-'}"


def load(src: Path) -> tuple[dict, list]:
    miss = [rel for rel in FILES.values() if not (src / rel).is_file()]
    if miss:
        BF.refuse(f"--source {src} lacks the figure build's file(s) {miss}")
    d = {k: pd.read_csv(src / rel, dtype={"seed": str}) for k, rel in FILES.items()}
    files = [dict(name=rel, sha256=BF.sha256(src / rel)) for rel in FILES.values()]
    return d, files


def objects(null: pd.DataFrame, cos: dict) -> list:
    """The 14 objects: the LoRAs by the null file's sha256 per measured object, the ReLU steers by the committed cosine data's."""
    objs = []
    for (o, s), g in null.groupby(["obj", "seed"], sort=True):
        shas = g.sha256.unique()
        if len(shas) != 1:
            BF.refuse(f"{FILES['null']}: {o} {s} names {len(shas)} objects")
        objs.append(dict(object_id=str(shas[0]), family="lora", role=o, seed="" if s == "-" else s.lstrip("s")))
    if cos.get("layer") != 12:
        BF.refuse(f"{COSINES.name}: layer {cos.get('layer')}, expected 12")
    for e in cos["entries"]:
        role = COS_LABEL.get(e["object_label"])
        if role is None or (role != "S" and e["fit_model_id"] != "qwen35_4b_base"):
            continue                                                # the instruct-direct fits are not in this figure
        objs.append(dict(object_id=e["object_id"], family="relu", role=role, seed="" if role == "S" else e["seed"].lstrip("s")))
    want = {(f, r, s) for f in ("relu", "lora") for r in ("C", "K") for s in SEEDS} | {("relu", "S", ""), ("lora", "S", "")}
    have = [(o["family"], o["role"], o["seed"]) for o in objs]
    if sorted(have) != sorted(want) or len({o["object_id"] for o in objs}) != len(objs):
        BF.refuse(f"objects: {sorted(have)}, expected one distinct sha256 for each of {sorted(want)}")
    for o in objs:
        o["key"] = key(o["role"], o["family"], o["seed"])
    return sorted(objs, key=lambda o: (o["family"], o["role"], o["seed"]))


def lora_block(pairs: pd.DataFrame, keys: set) -> dict:
    a = pairs[pairs.panel.eq("a")]
    n = a.groupby(["space", "layer", "series"]).size()
    want = {(sp, l, se): k for sp in ("write", "read") for l in LORA_LAYERS for se, k in LORA_SERIES.items()}
    if n.to_dict() != want:
        BF.refuse(f"{FILES['pairs']} panel a: pairs per space x layer x series {n.to_dict()}, expected {want}")
    if not set(a.measured) | set(a.span_of) <= keys:
        BF.refuse(f"{FILES['pairs']} panel a names objects outside the pinned set: {sorted(set(a.measured) | set(a.span_of) - keys)}")
    if not a.value.between(0, 1).all():
        BF.refuse(f"{FILES['pairs']} panel a: a share outside [0, 1]")
    a = a.sort_values(["space", "layer", "series", "measured", "span_of"])
    return BF.block(["space", "layer", "series", "measured", "span_of", "pair_type", "share"],
                    [[r.space, int(r.layer), r.series, r.measured, r.span_of, r.pair_type, float(r.value)] for r in a.itertuples()])


def relu_block(pairs: pd.DataFrame, keys: set) -> dict:
    b = pairs[pairs.panel.eq("b")]
    n = b.groupby(["space", "layer", "series"]).size()
    want = {(sp, l, se): k for sp in ("read", "write") for l in range(N_LAYERS) for se, k in RELU_SERIES.items()}
    if n.to_dict() != want:
        BF.refuse(f"{FILES['pairs']} panel b: pairs per vector x layer x series {n.to_dict()}, expected {want}")
    if not set(b.measured) | set(b.span_of) <= keys:
        BF.refuse(f"{FILES['pairs']} panel b names objects outside the pinned set: {sorted(set(b.measured) | set(b.span_of) - keys)}")
    if not b.value.between(-1, 1).all():
        BF.refuse(f"{FILES['pairs']} panel b: a cosine outside [-1, 1]")
    b = b.sort_values(["space", "layer", "series", "measured", "span_of"])
    return BF.block(["vector", "layer", "series", "a", "b", "cosine"],
                    [[r.space, int(r.layer), r.series, r.measured, r.span_of, float(r.value)] for r in b.itertuples()])


def null_block(null: pd.DataFrame, objs: list) -> dict:
    lora = {(o["role"], o["seed"]): o["object_id"] for o in objs if o["family"] == "lora"}
    if set(null.n_null) != {N_NULL} or set(null.null_seed) != {NULL_SEED}:
        BF.refuse(f"{FILES['null']}: draws {sorted(set(null.n_null))}, seeds {sorted(set(null.null_seed))}; expected {N_NULL} draws, seed {NULL_SEED}")
    rows = []
    for r in null.sort_values(["obj", "seed", "space", "layer"]).itertuples():
        seed = "" if r.seed == "-" else r.seed.lstrip("s")
        if lora.get((r.obj, seed)) != r.sha256:
            BF.refuse(f"{FILES['null']}: {r.obj} {r.seed} is {r.sha256[:12]}, not the pinned LoRA")
        exp = RANK / (D_MODEL if r.space == "write" else D_IN)
        if not math.isclose(float(r.expected), exp, rel_tol=1e-9):
            BF.refuse(f"{FILES['null']}: {r.obj} {r.seed} layer {r.layer} {r.space}: expected share {r.expected}, not {exp}")
        rows.append([key(r.obj, "lora", seed), int(r.layer), r.space, float(r.mean), float(r.q025), float(r.q50), float(r.q975), float(r.q99)])
    want = {(k, l, sp) for k in (key(o, "lora", s) for o, s in lora) for l in LORA_LAYERS for sp in ("write", "read")}
    if {(r[0], r[1], r[2]) for r in rows} != want or len(rows) != len(want):
        BF.refuse(f"{FILES['null']}: every LoRA x layer x space once")
    return {"n_draws": N_NULL, "seed": NULL_SEED, "expected": {"write": RANK / D_MODEL, "read": RANK / D_IN},
            **BF.block(["measured", "layer", "space", "mean", "q0.025", "q0.5", "q0.975", "q0.99"], rows)}


def check_layer12(relu: dict, objs: list, cos: dict) -> int:
    """The layer-12 cosines equal the committed cosine data's, value for value; returns how many were compared."""
    rc = pd.DataFrame(relu["rows"], columns=relu["columns"])
    rc = rc[rc.layer.eq(12)].set_index(["vector", "a", "b"]).cosine
    n = 0
    for e in cos["cross_pairs"] + cos["pairs"]:
        oa, ob = e.get("object_a", e.get("object_label")), e.get("object_b", e.get("object_label"))
        if oa not in COS_LABEL or ob not in COS_LABEL:
            continue
        a, b = key(COS_LABEL[oa], "relu", e["seed_a"].lstrip("s")), key(COS_LABEL[ob], "relu", e["seed_b"].lstrip("s"))
        for vec in ("read", "write"):
            if (vec, a, b) not in rc.index or not math.isclose(rc[(vec, a, b)], e[vec], rel_tol=LAYER12_REL, abs_tol=1e-9):
                BF.refuse(f"layer-12 {vec} cosine {a} vs {b}: {rc.get((vec, a, b))!r}, {COSINES.name} says {e[vec]!r}")
            n += 1
    if n != 30:
        BF.refuse(f"{COSINES.name}: {n} layer-12 values compared, expected 30 (9 cross pairs and 6 seed pairs, read and write)")
    return n


def check_values(sc: dict, values: pd.DataFrame) -> float:
    """The drawn means, ranges and bands recomputed from the sidecar (the reader's own rule) against the figure build's values CSV,
    within the two files' printed precision; returns the largest absolute difference."""
    sys.path.insert(0, str(HERE.parent))
    import appendix_geometry_overlap as GO                         # the reader the notebook draws with
    got = pd.concat([GO.summary_frame(sc).assign(drawn=True),
                     GO.band_frame(sc).rename(columns={"lo": "min", "hi": "max", "null_mean": "mean"}).assign(panel="a", series="random")],
                    ignore_index=True)
    v = values[~(values.panel.eq("b") & values.series.eq("random"))].set_index(["panel", "space", "series", "layer"])
    g = got.set_index(["panel", "space", "series", "layer"])
    if set(v.index) != set(g.index):
        BF.refuse(f"{FILES['values']}: drawn keys differ from the recomputation: {sorted(set(v.index) ^ set(g.index))[:5]}")
    worst = 0.0
    for col in ("mean", "min", "max"):
        a, b = v[col].astype(float).values, g.loc[v.index, col].astype(float).values
        if not np.allclose(a, b, rtol=REL, atol=ABS):
            i = int(np.argmax(np.abs(a - b) - ABS - REL * np.abs(b)))
            BF.refuse(f"{FILES['values']}: the recomputed {col} differs from the build's at {v.index[i]}: {a[i]!r} vs {b[i]!r}")
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


def build(src: Path, db: Path) -> dict:
    d, files = load(src)
    cos = json.loads(COSINES.read_text())
    objs = objects(d["null"], cos)
    head = BF.bind_registry(objs, db)
    keys = {o["key"] for o in objs}
    relu = relu_block(d["pairs"], keys)
    n12 = check_layer12(relu, objs, cos)
    out = {
        "schema": SCHEMA,
        "about": ABOUT,
        "judge": JUDGE,
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": SCRIPT, "script_sha256": BF.sha256(Path(__file__)),
            "git_commit": BF.git("rev-parse", "HEAD"), "git_dirty": bool(BF.git("status", "--porcelain", "--", SCRIPT)),
            "figure_build": ("the geometry analysis's figure build, outside this repository; its files are named relative to its output "
                             "directory, each with its sha256. Panel (a)'s shares are the weight lane's one-sided overlaps, recomputed "
                             "there from the object files; panel (b)'s cosines were computed there from the object files"),
            "checks": {"layer12_cosines_equal": f"{COSINES.name}, {n12} values, to 8 significant digits (the pairs file's precision)"},
            "registry": {"opened": "read-only", "head_at_build": head},
            "source_files": files + [dict(name=f"paper/notebooks/data/{COSINES.name}", sha256=BF.sha256(COSINES))],
        },
        "objects": [{k: o[k] for k in ("object_id", "key", "family", "role", "seed", "registry")} for o in objs],
        "d_model": D_MODEL, "d_in": D_IN, "rank": RANK,
        "lora_overlap": lora_block(d["pairs"], keys),
        "lora_null": null_block(d["null"], objs),
        "relu_cosine": relu,
    }
    out["provenance"]["checks"]["values_max_abs_diff"] = check_values(out, d["values"])
    return out


VOLATILE = BF.VOLATILE + ("values_max_abs_diff",)


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in VOLATILE}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=os.environ.get("APPENDIX_GEOMETRY_OVERLAP_SOURCE"))
    ap.add_argument("--registry", default=os.path.join(os.path.expanduser(os.environ.get("REGISTRY_ROOT", "~/registry")), "registry.sqlite"))
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the committed file")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if not a.source:
        BF.refuse("no --source (the figure build's output directory) and APPENDIX_GEOMETRY_OVERLAP_SOURCE is unset")
    out = build(Path(os.path.expanduser(a.source)), Path(os.path.expanduser(a.registry)))
    text = BF._dump(out) + "\n"
    json.loads(text)                                            # the writer's output parses back
    if a.check:
        old = json.loads(Path(a.out).read_text())
        new = json.loads(text)
        if _strip(old) != _strip(new):
            diff = [k for k in set(_strip(old)) | set(_strip(new)) if _strip(old).get(k) != _strip(new).get(k)]
            print(f"DIFFERS from {a.out}: top-level keys {sorted(diff)}")
            return 1
        print(f"{a.out}: matches a fresh build ({len(out['objects'])} objects, {len(out['provenance']['source_files'])} source files)")
        return 0
    Path(a.out).write_text(text)
    print(f"wrote {a.out} ({len(text) / 1e3:.0f} kB): {len(out['objects'])} objects bound to the registry (head "
          f"{out['provenance']['registry']['head_at_build']}); {len(out['lora_overlap']['rows'])} LoRA overlaps, "
          f"{len(out['relu_cosine']['rows'])} ReLU cosines, {len(out['lora_null']['rows'])} null rows; the drawn values match the "
          f"build's to {out['provenance']['checks']['values_max_abs_diff']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
