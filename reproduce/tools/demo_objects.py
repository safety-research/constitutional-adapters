#!/usr/bin/env python3
"""reproduce/tools/demo_objects.py — one training seed for every fitted Qwen3.5-4B intervention, written to objects/DEMO.json.

    python reproduce/tools/demo_objects.py            # recompute and write objects/DEMO.json
    python reproduce/tools/demo_objects.py --check    # recompute and compare with the committed file (exit 1 on a difference)

The deltas are the figure notebook's own. The tool executes the header and load cells of
paper/notebooks/registry_qwen_figures.ipynb as the notebook does, against the committed paper-frame snapshot
(REGISTRY_NB_SOURCE=snapshot), then the two calls the Figure 3 cell makes next — `R.anchored(op, pool=POOL_Q)` and
`R.category_means(d, categories=R.categories_for(R.MODEL_ID))` — so every number here went through the same readers,
anchors, exclusions and seed handling as Figure 4, and a change to the notebook's load chain changes this file with it.

The metric is the notebook's multi-turn category mean at the operating dose on the paper lane: for one training seed, the
mean over Crescendo and ActorAttack of the delta in defense success (100 − attack success rate at threshold 0.5) against
the unsteered model, in percentage points, positive being better defense. ONE training seed is chosen for all six fitted
interventions (control, constitution and difference, for the LoRA adapter and the ReLU steer): the seed whose difference
LoRA (`difference_lora`, the paper's headline intervention) has the highest value, compared unrounded (the file carries
every seed's values at full precision, so the choice can be checked from the file alone); ties go to the lower seed. Each
fitted intervention then contributes its fit at that seed, and the tool refuses by name when one has none. One seed keeps
the set self-consistent: each difference object is derived from the same seed's control and constitution fits, so the
four control and constitution objects are exactly the two differences' parents (`derived_meta.json` parent_pos /
parent_neg, one model key), which the tool verifies and the file records. The refusal-direction baseline has no training
seeds and contributes its operating object (c = 0.3). Object ids come from the frame and are cross-checked against
objects/MANIFEST.json through `sisteer.objects`.

Which lane the rows come from is the notebook's call, not this tool's: a fitted object's rows enter the paper lane as
`registry_nb.admit_lane` admits them (the roster's comparison ids, else the lane column). In the committed snapshot every
fitted Qwen3.5-4B object in Figures 3 to 6 is a qwen35_4b_base fit served on qwen35_4b_it (lane base_to_instruct), and the
instruct-direct fits, one per label with no training seeds, are retired from those panels; each entry records its `lane`.

`source.inputs_sha256` digests what the list is computed from (the notebook's code cells, registry_nb.py, the snapshot
reader, the snapshot's CURRENT and MANIFEST.json, objects/MANIFEST.json, this tool). tests/test_demo_objects.py compares
them with the tree on every run, so a change to any of those fails the ordinary tests until the list is recomputed; the
full recomputation behind `--check` is the slow, opt-in test.

Needs the notebook environment (`uv sync --frozen --extra cpu --group notebook`), `git` on PATH (the header cell prints
the code revision), and about 2 GB under paper/notebooks/_cache/ (gitignored) for the reassembled snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "paper" / "notebooks" / "registry_qwen_figures.ipynb"
OUT = REPO / "objects" / "DEMO.json"
SCHEMA = "demo_objects/2"
CATEGORY = "multi-turn"
#: The six fitted interventions and the baseline, in the notebook's roster order (registry_nb.OBJECTS) minus the prompts.
FITTED = ("control_relu", "control_lora", "constitution_relu", "constitution_lora", "difference_relu", "difference_lora")
BASELINE = "arditi_allpos"
DEMO_LABELS = (BASELINE,) + FITTED
#: The intervention whose best training seed is the one seed every fitted intervention takes.
SEED_ANCHOR = "difference_lora"
#: Each difference and the labels of the parents its derived_meta.json names: (parent_pos, parent_neg) = (constitution, control).
PARENTS = {"difference_lora": ("constitution_lora", "control_lora"), "difference_relu": ("constitution_relu", "control_relu")}
#: The Figure 3 cell's first lines: this tool runs exactly these after the load cell, and refuses when they change.
FIGURE_CALLS = ("d = R.anchored(op, pool=POOL_Q)", "CAT_Q = R.categories_for(R.MODEL_ID)",
                "cm = R.category_means(d, categories=CAT_Q)")
_SEED = re.compile(r"^s(\d+)$")
#: What the list is computed from, digested into `source.inputs_sha256`; `<CURRENT>` is the snapshot id paper/snapshot/CURRENT
#: names. The notebook is digested by its code cells' sources, so the figures stage, which rewrites the file without its
#: outputs, leaves the digest unchanged.
INPUTS = ("paper/notebooks/registry_qwen_figures.ipynb", "paper/notebooks/registry_nb.py", "src/sisteer/registry/snapshot.py",
          "src/sisteer/registry/paper.py", "paper/snapshot/CURRENT", "paper/snapshot/snapshots/<CURRENT>/MANIFEST.json",
          "objects/MANIFEST.json", "reproduce/tools/demo_objects.py")


def inputs_sha256(repo: Path = REPO) -> dict[str, str]:
    """sha256 of each input in INPUTS under `repo`, keyed by its path (the notebook's key ends in `#code`)."""
    current = (repo / "paper" / "snapshot" / "CURRENT").read_text(encoding="utf-8").strip()
    out = {}
    for rel in INPUTS:
        rel = rel.replace("<CURRENT>", current)
        p = repo / rel
        if rel.endswith(".ipynb"):
            with open(p, encoding="utf-8") as fh:
                nb = json.load(fh)
            data = json.dumps(["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]).encode("utf-8")
            rel += "#code"
        else:
            data = p.read_bytes()
        out[rel] = hashlib.sha256(data).hexdigest()
    return out


def _cell(cells: list[str], marker: str, what: str) -> str:
    hits = [c for c in cells if marker in c]
    if len(hits) != 1:
        sys.exit(f"REFUSED: {NOTEBOOK.relative_to(REPO)} has {len(hits)} code cells containing {marker!r}; "
                 f"expected exactly one ({what})")
    return hits[0]


def notebook_frames(snapshot_root: str | None = None) -> dict:
    """Run the notebook's header and load cells, then the Figure 3 cell's anchoring, in one namespace; returns it."""
    os.environ["REGISTRY_NB_SOURCE"] = "snapshot"
    os.environ.setdefault("MPLBACKEND", "Agg")
    if snapshot_root:
        os.environ["REGISTRY_NB_SNAPSHOT_ROOT"] = snapshot_root
    nbdir = str(NOTEBOOK.parent)
    os.chdir(nbdir)                                   # the header cell inserts os.getcwd() for `import registry_nb`
    sys.path.insert(0, nbdir)
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    header = _cell(cells, "import registry_nb as R", "the header cell")
    load = _cell(cells, "PAPER_SQL =", "the load cell")
    figure = _cell(cells, FIGURE_CALLS[0], "the Figure 3 cell")
    missing = [c for c in FIGURE_CALLS if c not in figure]
    if missing:
        sys.exit("REFUSED: the notebook's Figure 3 cell no longer computes its deltas the way this tool assumes "
                 f"(missing {missing}); update FIGURE_CALLS in {Path(__file__).relative_to(REPO)} to match")
    ns: dict = {"__name__": "__demo_objects__"}
    with contextlib.redirect_stdout(sys.stderr):      # the notebook's own report goes to stderr; stdout is this tool's
        exec(compile(header, "<notebook header cell>", "exec"), ns)
        exec(compile(load, "<notebook load cell>", "exec"), ns)
        exec(compile("\n".join(FIGURE_CALLS), "<notebook figure cell: anchoring>", "exec"), ns)
    return ns


def _r(x) -> float:
    """A frame value as a plain float, at full precision (JSON round-trips it exactly; nothing here is rounded)."""
    return float(x)


def _one(values, what: str):
    vals = sorted({str(v) for v in values})
    if len(vals) != 1:
        sys.exit(f"REFUSED: {what} is not unique in the frame: {vals}")
    return vals[0]


def per_eval(rows, evals: list, sign: int) -> dict:
    """Per eval, the delta the category mean averages (the draws' mean, oriented) and the reading behind it."""
    out = {}
    for ev in evals:
        g = rows[rows.eval_id == ev]
        if not len(g):
            sys.exit(f"REFUSED: no multi-turn row on {ev} for {_one(rows.object_label, 'label')} "
                     f"seed {_one(rows.seed, 'seed') or '-'}")
        n = _one(g.n.astype(int), f"{ev} n")
        rec = dict(gain_pp=_r(sign * g.d_y.mean()), asr=_r(g.y.mean()), anchor_asr=_r(g.anchor_value.mean()), n=int(n),
                   draws=int(len(g)), anchor_runs=int(g.anchor_draws.max()))
        if len(g) == 1:
            rec["k"] = int(g.k.iloc[0])
        out[ev] = rec
    return out


def shared_seed(candidates: dict[str, list[dict]]) -> int:
    """The one training seed: the seed of the SEED_ANCHOR candidate with the highest unrounded multi_turn_gain_pp, ties to
    the lower seed. `candidates` maps each demo label to its per-seed candidates (dicts with `seed` and `multi_turn_gain_pp`)."""
    anchor = candidates.get(SEED_ANCHOR) or []
    if not anchor:
        sys.exit(f"REFUSED: no {SEED_ANCHOR} candidate to choose the training seed from")
    if any(c["seed"] is None for c in anchor):
        sys.exit(f"REFUSED: a {SEED_ANCHOR} candidate has no training seed; the one seed is chosen among its seeds")
    best = min(anchor, key=lambda c: (-c["multi_turn_gain_pp"], c["seed"]))
    return int(best["seed"])


def choose(candidates: dict[str, list[dict]]) -> tuple[int, dict[str, dict]]:
    """(the one training seed, the chosen candidate per label): every fitted intervention's candidate at the seed
    `shared_seed` picks, refusing by name when one has no fit at it; the baseline's single operating object. Sets each
    candidate's `chosen` flag."""
    seed = shared_seed(candidates)
    chosen = {}
    for label, cands in candidates.items():
        if label in FITTED:
            hits = [c for c in cands if c["seed"] == seed]
            if len(hits) != 1:
                have = ", ".join(f"s{c['seed']}" for c in cands) or "none"
                sys.exit(f"REFUSED: {label} has {len(hits) or 'no'} candidate{'s' if len(hits) != 1 else ''} at training "
                         f"seed {seed} (its seeds: {have}); the one seed is {SEED_ANCHOR}'s best, and every fitted "
                         "intervention must carry a fit at it")
            pick = hits[0]
        else:
            if len(cands) != 1:
                sys.exit(f"REFUSED: {label} has {len(cands)} candidates; the refusal direction has one operating object")
            pick = cands[0]
        for c in cands:
            c["chosen"] = c is pick
        chosen[label] = pick
    return seed, chosen


def parents_are_the_chosen_fits(entries: list[dict], repo: Path = REPO) -> str:
    """Why one seed: each chosen difference's derived_meta.json must name the chosen constitution (parent_pos) and control
    (parent_neg) of its family, all under one model key. Records each difference's `parents` (label -> object id) on its
    entry and returns the model key; refuses by name otherwise."""
    by_label = {e["paper_label"]: e for e in entries}
    keys = {}
    for diff, (pos, neg) in PARENTS.items():
        dm = json.loads((repo / "objects" / by_label[diff]["dir"] / "derived_meta.json").read_text(encoding="utf-8"))
        have = tuple(dm[side]["object_sha256"] for side in ("parent_pos", "parent_neg"))
        want = (by_label[pos]["object_id"], by_label[neg]["object_id"])
        if have != want:
            sys.exit(f"REFUSED: {diff} {by_label[diff]['object_id'][:12]} is derived from {have[0][:12]} (pos) and "
                     f"{have[1][:12]} (neg), not the chosen {pos} {want[0][:12]} and {neg} {want[1][:12]}; one training seed "
                     "should make the four control and constitution objects exactly the two differences' parents")
        by_label[diff]["parents"] = {pos: want[0], neg: want[1]}
        keys[diff] = dm.get("model_key")
        for side in ("parent_pos", "parent_neg"):
            keys[f"{diff}.{side}"] = dm[side].get("model_key")
    if len(set(keys.values())) != 1 or None in keys.values():
        sys.exit(f"REFUSED: the two differences and their parents do not share one model key: {keys}")
    return next(iter(keys.values()))


def compute(ns: dict) -> dict:
    """The document: one entry per demo intervention, its object at the one training seed and every seed's candidate."""
    import pandas as pd                               # the notebook environment's; imported here so --help needs nothing
    from sisteer.objects import load_index, resolve

    R, d, cm = ns["R"], ns["d"], ns["cm"]
    evals, sign, _ylabel = R.CATEGORIES[CATEGORY]
    model = R.MODEL_ID
    mt_cm = cm[(cm.category == CATEGORY) & (cm.model_id == model)]
    mt_d = d[d.eval_id.isin(evals) & (d.model_id == model)]
    idx = load_index(REPO)
    # pass 1: every label's candidates, one per training seed (the baseline's one operating object), from the frame
    candidates: dict[str, list[dict]] = {}
    for label in DEMO_LABELS:
        c_rows = mt_cm[mt_cm.object_label == label]
        if not len(c_rows):
            sys.exit(f"REFUSED: the notebook's {CATEGORY} category means carry no {label} row at the operating dose")
        cands = []
        for _, c in c_rows.sort_values("seed").iterrows():
            seed_label = str(c.seed or "")
            m = _SEED.match(seed_label)
            if label in FITTED and not m:
                sys.exit(f"REFUSED: {label} carries a {CATEGORY} mean with seed {seed_label!r}; a fitted intervention's "
                         "rows are its training seeds (the canonical fit is retired by the notebook)")
            if label == BASELINE and seed_label:
                sys.exit(f"REFUSED: {label} carries seed {seed_label!r}; the refusal direction has no training seeds")
            rows = mt_d[(mt_d.object_label == label) & (mt_d.seed.fillna("").astype(str) == seed_label)]
            oid = _one(rows.object_id, f"{label} {seed_label or 'operating'} object_id")
            lane = _one(rows.lane, f"{label} {seed_label} lane")
            dose = float(_one(rows.dose.astype(float), f"{label} {seed_label} dose"))
            if bool(c.partial) or int(c.k) != len(evals):
                sys.exit(f"REFUSED: {label} {seed_label}: the {CATEGORY} mean spans {int(c.k)} of {len(evals)} evals")
            cands.append(dict(seed=int(m.group(1)) if m else None, object_id=oid, lane=lane, dose=_r(dose),
                              multi_turn_gain_pp=_r(c.y), lo=_r(c.lo), hi=_r(c.hi),
                              per_eval=per_eval(rows, list(evals), sign)))
        candidates[label] = cands
    # the one seed, and each label's candidate at it
    seed, chosen = choose(candidates)
    # pass 2: cross-check every chosen object against the manifest, the sidecars and the address resolver
    entries = []
    for label in DEMO_LABELS:
        best = chosen[label]
        e = idx.get(best["object_id"]) if best["object_id"] in idx._by_id else None
        if e is None:
            sys.exit(f"REFUSED: the frame's {label} object {best['object_id'][:12]} is not in objects/MANIFEST.json")
        if e.label != label:
            sys.exit(f"REFUSED: {best['object_id'][:12]} is {e.label!r} in the manifest, {label!r} in the frame")
        if label in FITTED:
            if not (abs(best["dose"] - 1.0) < 1e-9 and e.dose == 1.0):
                sys.exit(f"REFUSED: {label} seed {best['seed']} is not at the operating dose 1.0 (frame {best['dose']}, "
                         f"manifest {e.dose})")
            if e.seed != best["seed"]:
                sys.exit(f"REFUSED: {label} {best['object_id'][:12]}: the frame says seed {best['seed']}, the sidecars {e.seed}")
            back = resolve(e.model_id, label, seed=best["seed"], repo_root=REPO)
        else:
            if e.dose is None or abs(e.dose - best["dose"]) > 1e-9:
                sys.exit(f"REFUSED: {label} {best['object_id'][:12]}: the frame's coefficient {best['dose']} is not the "
                         f"sidecar's {e.dose}")
            back = resolve(e.model_id, label, dose=best["dose"], repo_root=REPO)
        if back != best["object_id"]:
            sys.exit(f"REFUSED: sisteer.objects resolves the {label} address to {back[:12]}, the frame drew {best['object_id'][:12]}")
        fit_model = _one(mt_d[mt_d.object_id == best["object_id"]].fit_model_id, f"{label} fit_model_id")
        if fit_model != e.model_id:
            sys.exit(f"REFUSED: {label}: the frame's fit model {fit_model} is not the manifest's {e.model_id}")
        entries.append(dict(paper_label=label, label=R.LABEL.get(label, label), object_id=best["object_id"],
                            model_id=e.model_id, dir=e.dir, object_kind=e.kind, derivation=e.derivation,
                            served_model_id=model, lane=best["lane"], dose=best["dose"], seed=best["seed"],
                            effective_batch=e.effective_batch, multi_turn_gain_pp=best["multi_turn_gain_pp"],
                            interval_95=[best["lo"], best["hi"]], candidates=candidates[label]))
    model_key = parents_are_the_chosen_fits(entries)
    fitted_lanes = sorted({c["lane"] for e in entries if e["paper_label"] in FITTED for c in e["candidates"]})
    baseline_lane = next(e["lane"] for e in entries if e["paper_label"] == BASELINE)
    return {
        "schema": SCHEMA,
        "title": ("One training seed for every fitted Qwen3.5-4B intervention: the seed of the best difference LoRA by "
                  "multi-turn defense gain"),
        "training_seed": seed,
        "seed_chosen_by": SEED_ANCHOR,
        "model_key": model_key,
        "metric": {
            "name": "multi_turn_gain_pp",
            "definition": ("The figure notebook's multi-turn category mean at the operating dose on the paper lane: the mean "
                           "over Crescendo and ActorAttack of the delta in defense success (100 - attack success rate at "
                           "threshold 0.5) against the unsteered model, in percentage points; positive is better defense. "
                           "Per eval, `gain_pp` is that delta (the draws' mean where an object has several decode draws), "
                           "`asr` the steered attack success rate, `anchor_asr` the unsteered one it is differenced against, "
                           "`n` the goals per draw, `k` the goals judged jailbroken, `anchor_runs` the unsteered runs pooled "
                           "into the anchor. Values are at full precision; the choice compares them unrounded."),
            "category": CATEGORY, "evals": list(evals), "orientation_sign": int(sign),
            "operating_dose": 1.0, "baseline_coefficient": next(x["dose"] for x in entries if x["paper_label"] == BASELINE),
            "served_model_id": model, "paper_lane": R.PAPER_LANE[model],
            "lane_note": (f"Every row is one the notebook admits to the paper lane of {model} ({R.PAPER_LANE[model]}) by "
                          "registry_nb.admit_lane (the roster's comparison ids, else the lane column); each entry's `lane` is "
                          "the lane its rows were recorded on, `model_id` the fit model and `served_model_id` the served one. "
                          f"In this snapshot the fitted interventions' training seeds are all {', '.join(fitted_lanes)} and "
                          f"the refusal direction is {baseline_lane}."),
        },
        "rule": (f"One training seed for every fitted intervention: the seed whose {SEED_ANCHOR} has the highest "
                 "multi_turn_gain_pp among the batch-64 replicates the figures average (ties to the lower seed), taken for "
                 "control, constitution and difference alike (`training_seed`; each candidate's `chosen` flag follows it). "
                 "One seed keeps the set self-consistent: each difference object is derived from the same seed's control "
                 "and constitution fits, so the four control and constitution objects are exactly the two differences' "
                 "parents (each difference's `parents`, from its derived_meta.json parent_pos / parent_neg), all under one "
                 "fit model key (`model_key`). The refusal direction has no training seeds and is its operating object."),
        "source": {"snapshot_id": R.SNAPSHOT_ID, "registry_head": int(ns["HEAD"]),
                   "notebook": str(NOTEBOOK.relative_to(REPO)),
                   "cells": ["header", "load", "figure 3: " + "; ".join(FIGURE_CALLS)],
                   "inputs_sha256": inputs_sha256(REPO)},
        "tool": str(Path(__file__).resolve().relative_to(REPO)) + " (--check recomputes and compares)",
        "objects": entries,
    }


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python reproduce/tools/demo_objects.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="recompute and compare with the committed file; exit 1 on a difference")
    ap.add_argument("--out", default=str(OUT), help=f"where to write (default {OUT.relative_to(REPO)})")
    ap.add_argument("--snapshot-root", default=None, help="REGISTRY_NB_SNAPSHOT_ROOT (default: the clone's paper/snapshot)")
    a = ap.parse_args(argv)
    doc = compute(notebook_frames(a.snapshot_root))
    text = dumps(doc)
    out = Path(a.out)
    if a.check:
        if not out.is_file():
            print(f"REFUSED: {out} does not exist; run without --check to write it", file=sys.stderr)
            return 1
        have = out.read_text(encoding="utf-8")
        if have == text:
            print(f"{out.relative_to(REPO) if out.is_relative_to(REPO) else out} is current: {_summary(doc)}")
            return 0
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), text.splitlines(True), "committed", "recomputed"))
        print(f"{out} differs from the recomputation (diff above)", file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}: {_summary(doc)}")
    return 0


def _summary(doc: dict) -> str:
    """One line: the one training seed and who chose it, then each entry's object, seed and gain."""
    return (f"training seed {doc['training_seed']} ({doc['seed_chosen_by']}'s best, model key {doc['model_key']}); "
            + ", ".join(f"{e['paper_label']} {e['object_id'][:12]} seed {e['seed'] if e['seed'] is not None else '-'} "
                        f"({e['multi_turn_gain_pp']:+.2f} pp)" for e in doc["objects"]))


if __name__ == "__main__":
    sys.exit(main())
