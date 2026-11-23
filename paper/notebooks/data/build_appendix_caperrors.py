"""Build `appendix_caperrors.json`, the sidecar behind the capability error analysis (appendix `app:caperrors`: Figures
`figA_qwen_capability_errors` and `figA_models_capability_errors`, Tables `tab:caperrors` and `tab:caperrors-bcb`).

    .venv/bin/python paper/notebooks/data/build_appendix_caperrors.py            # from the repository root: write the sidecar
    .venv/bin/python paper/notebooks/data/build_appendix_caperrors.py --check    # rebuild in memory and compare with the committed file

WHAT IT HOLDS. For every BFCL and BigCodeBench run the capability figures draw (Figure 6 and the all-model capability figure, registry
head 456957), the counts of NEW FAILURES (items the run's seed-matched unsteered run answered correctly and the run did not) by failure
class, the gains (the reverse), the BFCL calling counts, and for every pool the figures and tables draw the item-bootstrap intervals
the error analysis computed. Every run and object is pinned by its registry id. No LLM judge is involved anywhere: BFCL is scored by its
abstract-syntax-tree checker and BigCodeBench by its unit tests (the registry's per-item `acc`), and each failure is classed by
deterministic rules over the stored response, the parsed tool calls, the extracted program and the test output.

WHERE IT COMES FROM. The error analysis's per-item class tables under its work directory SOURCE, which is outside the repository
(--source, or APPENDIX_CAPERRORS_SOURCE; the script refuses without one): `bfcl/items_classified.parquet` (+ `items_meta.parquet`), `bcb/items_classified.parquet`, the
run manifest `scope/manifest.json` (run ids, object ids, anchors), and the interval tables `finalize/bfcl_split_pools.csv`,
`synth/synth_pools.csv`, `finalize/calling_ci.csv`, `synth/synth_premises.csv`, `synth/synth_anchor_churn.csv`. The class tables hold
response text and are not committed; this script reduces them to counts. The counts are recomputed here from the per-item classes with
the analysis's rules written out below (no code is imported from SOURCE), and every point value is checked against the analysis's own
tables to 1e-9 before anything is written: a rule copied wrongly refuses the build by name. Intervals are not recomputed (an item
bootstrap needs the per-item vectors); they are copied from the analysis's tables, B = 4000 item resamples with seed 20261111.

THE BINDING. Every run's correct count, its new failures and its gains are recomputed from the committed paper snapshot's per-item
results (`paper/snapshot/.../item_result.parquet`) against its anchors and must match, and every run's object id must be the one the
snapshot's run table names; the notebook repeats the same check against whichever source it reads (`appendix_caperrors.item_check`).

The script reads; it writes only the sidecar (and nothing with --check). It makes no network or API call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "appendix_caperrors.json"
SCHEMA = "appendix-caperrors-v1"
SOURCE_ENV = "APPENDIX_CAPERRORS_SOURCE"
SNAPSHOT_ROOT = REPO / "paper" / "snapshot"
TOL = 1e-9

# the source files the sidecar is built from, relative to SOURCE; each is hashed into the provenance block
FILES = {
    "manifest": "scope/manifest.json",
    "bfcl_items": "bfcl/items_classified.parquet",
    "bfcl_meta": "bfcl/items_meta.parquet",
    "bcb_items": "bcb/items_classified.parquet",
    "bfcl_pools": "finalize/bfcl_split_pools.csv",
    "pools": "synth/synth_pools.csv",
    "calling": "finalize/calling_ci.csv",
    "premises": "synth/synth_premises.csv",
    "churn": "synth/synth_anchor_churn.csv",
}
# the analysis code whose rules this script restates (hashed so a later change to a rule is visible in the provenance; not imported)
RULE_FILES = {
    "flip_engine": "synth/compute.py",
    "bfcl_declination_split": "finalize/bfcl_split.py",
    "declination_regexes": "finalize/recheck.py",
    "calling_intervals": "finalize/calling_ci.py",
    "bfcl_run_anchors": "bfcl/runs.py",
}

MODEL_OF = {"qwen35_4b_it": "qwen", "gemma4e4b_it": "gemma", "nemotron3nano4b": "nemotron", "llama8b_it": "llama"}
MODEL_ORDER = ["qwen35_4b_it", "gemma4e4b_it", "nemotron3nano4b", "llama8b_it"]
EVALS = {"bfcl": "bfcl", "bigcodebench": "bcb"}            # registry eval id -> the analysis's eval key
OBJECT_OF_ARM = {"refusal_steer": "arditi_allpos"}          # the analysis's arm name -> the registry's paper label (the rest are equal)
FITTED = ["control_relu", "constitution_relu", "difference_relu", "control_lora", "constitution_lora", "difference_lora"]
PROMPTED = ["safety_si", "constitution_si", "arditi_allpos"]
FAMILIES = {"all_relu": FITTED[:3], "all_lora": FITTED[3:], "all_fitted": FITTED}
POOL_OF_FAMILY = {"all_relu": "POOLED_relu", "all_lora": "POOLED_lora", "all_fitted": "POOLED_fitted"}

# THE CLASSES, in the figures' stacking order, each with the analysis's segment name. A new failure takes exactly one class.
CLASSES = {
    "bfcl": [
        ("no_call_other", "prose_other", "no tool call: an answer in prose, a clarifying question or a claim of no access"),
        ("no_call_declination", "decl", "no tool call, with explicit declination wording (keyword rule)"),
        ("malformed_call", "F_malformed", "an attempted call the parser could not extract intact"),
        ("call_then_cap", "F_runaway", "a parsed call followed by text that ran to the 1,024-token cap"),
        ("wrong_call", "WRONG", "an intact call with the wrong function, arguments or number of calls"),
        ("irrelevance_call", "DECISION", "a call on an irrelevance item, where the correct answer is to call nothing"),
    ],
    "bigcodebench": [
        ("no_code_prose", "F_prose", "format: prose and no code, not a refusal"),
        ("refusal_or_objection", "BEHAV", "a refusal of or objection to the task (keyword rule over the text before the code)"),
        ("malformed_code", "F_malformed", "format: a malformed or mis-extracted program"),
        ("cap_no_program", "F_runaway", "format: ran to the 4,096-token cap with no complete program"),
        ("wrong_code", "WRONG", "wrong code: a logic, API or syntax error in the extracted program"),
        ("undecidable", "AMBIG", "the answer fails only a self-test it appended itself"),
    ],
}
# further counts over the new failures (not a partition): name, the analysis's label, definition
EXTRAS = {
    "bfcl": [
        ("no_call", "F_prose", "no tool call at all (no_call_other + no_call_declination)"),
        ("declination_narrow", "ref_decl2_nocall", "no tool call and the narrower declination rule (the refute verifier's regexes)"),
        ("truncated", "truncated", "reached the 1,024-token cap"),
        ("truncated_loop", "trunc_loop", "reached the cap in a loop repeating a passage"),
    ],
    "bigcodebench": [
        ("refusal", "refusal", "an outright refusal"),
        ("refusal_safety", "refusal_safety", "an outright refusal on safety grounds"),
        ("objection_extraction", "objection_extraction", "an objection that left no extractable program"),
        ("objection_code", "objection_code", "code that fails after a preamble disputing the specification"),
        ("header_omitted", "header_omitted", "format: a bare function without the imports the task header supplies"),
        ("truncated", "truncated", "reached the 4,096-token cap"),
    ],
}
# composite shares the analysis reports with their own intervals: name, the analysis's label, the classes they sum
COMPOSITES = {
    "bfcl": [("format", "FORMAT", None), ("no_intact_call", "NO_INTACT_CALL", None)],
    "bigcodebench": [("format", "FORMAT", ["no_code_prose", "malformed_code", "cap_no_program"])],
}
COMPOSITE_PARTS = {"bfcl": {"format": ["no_call_other", "no_call_declination", "malformed_call", "call_then_cap"],
                            "no_intact_call": ["no_call_other", "no_call_declination", "malformed_call"]},
                   "bigcodebench": {"format": ["no_code_prose", "malformed_code", "cap_no_program"]}}
CALLING = [("relevant_no_call", "items that require a call answered with no parsed call", "n_relevant"),
           ("irrelevance_called", "irrelevance items answered with a call", "n_irrelevance")]

# THE DECLINATION RULES (BFCL), restated from finalize/recheck.py: `decl_flags` (M_OPEN | M_WONT | M_SELF on quote-normalised text) is
# the declination class; the refute verifier's RX_START | RX_NG on the raw text is the narrower reading carried beside it.
M_OPEN = re.compile(r"^\W*(?:I can'?t do that|I can'?t help with (?:that|this)|I won'?t do that|I'?m not going to do that|"
                    r"I'?m not able to do that|I can'?t assist with (?:that|this)|I'?m not going to help with)", re.I)
M_WONT = re.compile(r"\bI'?m not going to (?:call|use|run|execute|invoke|make|do|generate|produce|output|give|provide|tell|"
                    r"compute|calculate|pretend|help|answer|look|search|fetch|retrieve|send|book|delete|create|perform|"
                    r"process|submit|guess|fabricate|invent|hand|trigger|fire|place|transfer|schedule|report|return|fill)\b|"
                    r"\bI won'?t (?:call|use|run|execute|invoke|do|make|pretend|fabricate|guess|help)\b|"
                    r"\bI(?:'?m| am) declining\b|\bI decline\b|\bdeclining to\b", re.I)
M_SELF = re.compile(r"rather than just (?:saying no|declin\w+|refus\w+)|what I'?m declining|\bmy refusal\b", re.I)
RX_START = re.compile(r"^\s*(I can'?t do that|I can’t do that|I won'?t do that|I'?m not going to do that|I can'?t help with that|"
                      r"I'?m not able to do that)", re.I)
RX_NG = re.compile(r"I'?m not going to (call|use|run|execute|invoke|make|do|generate|produce|output|give|provide|tell|compute|"
                   r"calculate|pretend)|I’m not going to|I won'?t (call|use|run|execute|invoke)|I'?m declining|I am declining|"
                   r"I decline|declining to", re.I)


def _norm(s: str) -> str:
    return (s or "").replace("’", "'").replace("‘", "'")


def declination(text: str) -> bool:
    t = _norm(text)
    return bool(M_OPEN.search(t) or M_WONT.search(t) or M_SELF.search(t))


def declination_narrow(text: str) -> bool:
    return bool(RX_START.search(text or "") or RX_NG.search(text or ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display(path) -> str:
    p = str(path)
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


class BuildError(RuntimeError):
    """A source the sidecar cannot be built from, named."""


# ---- the runs ---------------------------------------------------------------------------------------------------------------------
def scope(manifest: dict) -> tuple:
    """(drawn, anchors): the runs the capability figures draw on their paper lanes, and the unsteered runs they are differenced against.
    A drawn run is a manifest run with role `steered`, not a set-aside duplicate and not an instruct-direct LoRA (those draw only on the
    transfer figure, against the mean of every unsteered run). Its anchor is the one seed-matched unsteered run its delta draws name."""
    runs = manifest["runs"]
    by_id = {r["run_id"]: r for r in runs}
    drawn, anchor_ids = [], set()
    for r in runs:
        if r["eval_id"] not in EVALS or "steered" not in r["roles"] or "set_aside_duplicate" in r["roles"]:
            continue
        if r["eval_id"] == "bfcl":
            sets = {tuple(sorted(d["anchor_run_ids"])) for d in r.get("draws", []) if "delta_pp" in d}
            if len(sets) != 1:
                raise BuildError(f"{r['run_id']}: {len(sets)} drawn anchor sets in the manifest, expected one")
            anchors = list(next(iter(sets)))
        else:
            a = r["anchor_run_id"]
            anchors = a if isinstance(a, list) else [a]
        instruct_direct = (r["view"].endswith("_it") and r["eval_id"] == "bfcl") or (
            r["eval_id"] == "bigcodebench" and str(r.get("store_view", "")).endswith("f22it") and r["arm"] != "refusal_steer")
        if instruct_direct:
            continue
        if len(anchors) != 1:
            raise BuildError(f"{r['run_id']}: {len(anchors)} anchors on a paper-lane run, expected the one seed-matched unsteered run")
        a = by_id.get(anchors[0])
        if a is None or a["arm"] != "unsteered" or a["eval_id"] != r["eval_id"] or a["model_id"] != r["model_id"]:
            raise BuildError(f"{r['run_id']}: anchor {anchors[0]} is not an unsteered run of the same model and eval in the manifest")
        drawn.append(r | {"_anchor": anchors[0]})
        anchor_ids.add(anchors[0])
    keys = [(r["model_id"], r["eval_id"], OBJECT_OF_ARM.get(r["arm"], r["arm"]), r.get("seed") or "", r["view"]) for r in drawn]
    if len(set(keys)) != len(keys):
        raise BuildError("two drawn runs share (model, eval, object, seed, view)")
    return drawn, [by_id[a] for a in sorted(anchor_ids)]


def _view_suffix(view: str) -> str:
    return view.rsplit("_", 1)[1]


ARM_OF_OBJECT = {v: k for k, v in OBJECT_OF_ARM.items()}


def bfcl_key(r: dict) -> str:
    """The BFCL class table's run key, `<model>:<arm>:<view suffix>` (bfcl/runs.py), for a manifest run or a sidecar run record."""
    arm = r["arm"] if "arm" in r else ARM_OF_OBJECT.get(r["object_label"], r["object_label"])
    return f"{MODEL_OF[r['model_id']]}:{arm}:{_view_suffix(r['view'])}"


def load_bfcl(src: Path, runs: list) -> tuple:
    """{run_id: per-item arrays aligned to the sorted item ids}, the item ids, the item categories. Classes per compute.load_bfcl and
    bfcl_split.py: FORMAT x format_part_strict (malformed / prose / runaway), CONTENT -> wrong call, DECISION -> irrelevance call; a
    no-call prose failure is a declination when `declination` matches its text."""
    meta = pd.read_parquet(src / FILES["bfcl_meta"], columns=["item_id", "category"])
    items = np.array(sorted(meta.item_id))
    cats = meta.set_index("item_id").loc[items].category.values
    cols = ["run_key", "item_id", "value", "group", "format_part_strict", "has_call", "relevant", "outcome", "trunc_kind", "content_log"]
    d = pd.read_parquet(src / FILES["bfcl_items"], columns=cols)
    frames = {k: g.set_index("item_id").loc[items] for k, g in d.groupby("run_key")}
    out = {}
    for r in runs:
        key = bfcl_key(r)
        if key not in frames:
            raise BuildError(f"{r['run_id']}: no rows under run key {key} in {FILES['bfcl_items']}")
        g = frames[key]
        v = g.value.values.astype(float)
        grp, fps = g.group.values, g.format_part_strict.values
        seg = np.full(len(items), "", dtype=object)
        for part in ("malformed", "prose", "runaway"):
            seg[(grp == "FORMAT") & (fps == part)] = "F_" + part
        seg[grp == "CONTENT"] = "WRONG"
        seg[grp == "DECISION"] = "DECISION"
        fail = v == 0
        if (seg[fail] == "").any() or (seg[~fail] != "").any():
            raise BuildError(f"{r['run_id']}: a failure without a class, or a correct item with one")
        text = g.content_log.fillna("").values
        has_call = g.has_call.values.astype(bool)
        decl = np.array([declination(x) for x in text])
        narrow = np.array([declination_narrow(x) for x in text])
        prose = fail & (seg == "F_prose")
        lab = {"prose_other": prose & ~decl, "decl": prose & decl, "F_malformed": seg == "F_malformed", "F_runaway": seg == "F_runaway",
               "WRONG": seg == "WRONG", "DECISION": seg == "DECISION", "F_prose": seg == "F_prose",
               "ref_decl2_nocall": fail & ~has_call & narrow, "truncated": fail & (g.outcome.values == "TRUNCATED"),
               "trunc_loop": fail & (g.outcome.values == "TRUNCATED") & (g.trunc_kind.values == "loop")}
        out[r["run_id"]] = {"v": v, "lab": lab, "relevant": g.relevant.values.astype(bool), "has_call": has_call}
    return out, items, cats


def load_bcb(src: Path, runs: list) -> dict:
    """{run_id: per-item arrays, items 0..1139}. Classes per compute.load_bcb: cause FORMAT splits into no code (NO_CODE), cap with no
    program (TRUNCATED / no_program) and malformed (every other FORMAT); CODE and SYNTAX_GENUINE are wrong code; REFUSAL and the two
    OBJECTION causes are refusal or objection; AMBIGUOUS_SELFTEST is undecidable."""
    cols = ["run_id", "item_id", "value", "cls", "sub", "cause", "refusal_grounds", "stop_reason"]
    d = pd.read_parquet(src / FILES["bcb_items"], columns=cols)
    frames = {rid: g.set_index("item_id").sort_index() for rid, g in d.groupby("run_id")}
    out = {}
    for r in runs:
        g = frames.get(r["run_id"])
        if g is None or list(g.index) != list(range(1140)):
            raise BuildError(f"{r['run_id']}: not the 1,140 BigCodeBench items in {FILES['bcb_items']}")
        cause, cls, sub = g.cause.values, g.cls.values, g["sub"].values
        v = g.value.values.astype(float)
        seg = np.full(1140, "", dtype=object)
        fmt = cause == "FORMAT"
        seg[fmt & (cls == "NO_CODE")] = "F_prose"
        seg[fmt & (cls == "TRUNCATED") & (sub == "no_program")] = "F_runaway"
        seg[fmt & (seg == "")] = "F_malformed"
        seg[np.isin(cause, ["CODE", "SYNTAX_GENUINE"])] = "WRONG"
        seg[np.isin(cause, ["REFUSAL", "OBJECTION_EXTRACTION", "OBJECTION_CODE"])] = "BEHAV"
        seg[cause == "AMBIGUOUS_SELFTEST"] = "AMBIG"
        fail = v == 0
        if (seg[fail] == "").any() or (seg[~fail] != "").any():
            raise BuildError(f"{r['run_id']}: a failure without a class, or a correct item with one")
        lab = {s: seg == s for s in ("F_prose", "BEHAV", "F_malformed", "F_runaway", "WRONG", "AMBIG")}
        lab.update({"refusal": fail & (cause == "REFUSAL"),
                    "refusal_safety": fail & (cause == "REFUSAL") & (g.refusal_grounds.values == "refusal_safety"),
                    "objection_extraction": fail & (cause == "OBJECTION_EXTRACTION"), "objection_code": fail & (cause == "OBJECTION_CODE"),
                    "header_omitted": fail & (cls == "EXTRACTION") & (sub == "header_omitted") & fmt,
                    "truncated": fail & (g.stop_reason.values == "max_tokens")})
        out[r["run_id"]] = {"v": v, "lab": lab}
    return out


def _count(x) -> int:
    """An integer count: every drawn run has one anchor, so each lost / gained sum is a whole number of items."""
    f = float(x)
    if abs(f - round(f)) > 1e-9:
        raise BuildError(f"count {f} is not a whole number")
    return int(round(f))


def run_counts(r: dict, arr: dict, anchor: dict, ev: str) -> dict:
    vs, va = arr["v"], anchor["v"]
    lost_w, gained_w = (1 - vs) * va, vs * (1 - va)
    cls = {name: _count((arr["lab"][src] * lost_w).sum()) for name, src, _ in CLASSES[ev]}
    ext = {name: _count((arr["lab"][src] * lost_w).sum()) for name, src, _ in EXTRAS[ev]}
    gained_cls = {name: _count((vs * (1 - va) * anchor["lab"][src]).sum()) for name, src, _ in CLASSES[ev]}
    out = {"k": _count(vs.sum()), "n": len(vs), "anchor_k": _count(va.sum()), "lost": _count(lost_w.sum()), "gained": _count(gained_w.sum()),
           "lost_by_class": cls, "lost_extra": ext, "gained_by_class": gained_cls}
    if sum(cls.values()) != out["lost"]:
        raise BuildError(f"{r['run_id']}: classes sum to {sum(cls.values())}, new failures {out['lost']}")
    if sum(gained_cls.values()) != out["gained"]:
        raise BuildError(f"{r['run_id']}: gained classes sum to {sum(gained_cls.values())}, gains {out['gained']}")
    if ev == "bfcl":
        rel = arr["relevant"]
        if not (rel == anchor["relevant"]).all():
            raise BuildError(f"{r['run_id']}: the run and its anchor disagree on which items require a call")
        out["calling"] = {"n_relevant": int(rel.sum()), "n_irrelevance": int((~rel).sum()),
                          "relevant_no_call": int((rel & ~arr["has_call"]).sum()), "irrelevance_called": int((~rel & arr["has_call"]).sum()),
                          "anchor_relevant_no_call": int((rel & ~anchor["has_call"]).sum()),
                          "anchor_irrelevance_called": int((~rel & anchor["has_call"]).sum())}
    return out


# ---- the pools --------------------------------------------------------------------------------------------------------------------
def pools(runs: list) -> list:
    """The pools the figures and tables draw, per compute.pools_for (paper lanes only): per model and object its drawn runs (Qwen's
    three seeds; one run elsewhere); per model the three families (all ReLU steers, all LoRAs, all fitted objects); across models
    each family, run-weighted and model-balanced (each model's runs weighted 1 / its run count, renormalised to the run count)."""
    out = []
    for ev in EVALS:
        rs = [r for r in runs if r["eval_id"] == ev]
        for m in MODEL_ORDER:
            mr = [r for r in rs if r["model_id"] == m]
            for o in FITTED + PROMPTED:
                ids = sorted(r["run_id"] for r in mr if r["object_label"] == o)
                if ids:
                    out.append(dict(eval_id=ev, model_id=m, pool=o, weighting="runs", run_ids=ids))
            for fam, objs in FAMILIES.items():
                ids = sorted(r["run_id"] for r in mr if r["object_label"] in objs)
                if ids:
                    out.append(dict(eval_id=ev, model_id=m, pool=fam, weighting="runs", run_ids=ids))
        for fam, objs in FAMILIES.items():
            ids = sorted(r["run_id"] for r in rs if r["object_label"] in objs)
            for w in ("runs", "model_balanced"):
                out.append(dict(eval_id=ev, model_id="all", pool=fam, weighting=w, run_ids=ids))
    return out


def pool_weights(run_ids: list, model_of: dict, weighting: str) -> dict:
    if weighting == "runs":
        return {r: 1.0 for r in run_ids}
    per = {}
    for r in run_ids:
        per.setdefault(model_of[r], []).append(r)
    tot = sum(1.0 / len(per[model_of[r]]) for r in run_ids)
    return {r: (1.0 / len(per[model_of[r]])) * len(run_ids) / tot for r in run_ids}


def pool_point(p: dict, by_id: dict) -> dict:
    """The pool's point values from its runs' counts: lost, gained, delta (pp), each class / extra / composite as % of the new failures,
    and the BFCL calling rates (% of the pooled items of each kind)."""
    ev = p["eval_id"]
    w = pool_weights(p["run_ids"], {r: by_id[r]["model_id"] for r in p["run_ids"]}, p["weighting"])
    rs = [by_id[r] for r in p["run_ids"]]
    lost = sum(w[r["run_id"]] * r["lost"] for r in rs)
    gained = sum(w[r["run_id"]] * r["gained"] for r in rs)
    n_runs = sum(w.values())
    n = rs[0]["n"]
    pt = {"lost": lost, "gained": gained, "delta_pp": 100.0 * (gained - lost) / (n * n_runs)}
    share = lambda c: 100.0 * c / lost if lost else math.nan  # noqa: E731
    pt["shares"] = {name: share(sum(w[r["run_id"]] * r["lost_by_class"][name] for r in rs)) for name, _s, _d in CLASSES[ev]}
    pt["shares"].update({name: share(sum(w[r["run_id"]] * r["lost_extra"][name] for r in rs)) for name, _s, _d in EXTRAS[ev]})
    pt["shares"].update({name: sum(pt["shares"][c] for c in parts) for name, parts in COMPOSITE_PARTS[ev].items()})
    if ev == "bfcl" and p["weighting"] == "runs":
        pt["calling"] = {}
        for name, _d, den in CALLING:
            num = sum(r["calling"][name] for r in rs)
            anum = sum(r["calling"]["anchor_" + name] for r in rs)
            dd = sum(r["calling"][den] for r in rs)
            pt["calling"][name] = 100.0 * num / dd
            pt["calling"]["anchor_" + name] = 100.0 * anum / dd
    return pt


# ---- the analysis's tables: point checks and intervals ----------------------------------------------------------------------------
def _close(a, b) -> bool:
    if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
        return (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b)))
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))


def _f(x):
    """A finite float or None (JSON has no inf / nan)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def attach_intervals(pool_list: list, by_id: dict, tables: dict, problems: list) -> None:
    """Copy the analysis's interval for every pooled quantity, after checking its point value equals the one recomputed from counts."""
    bf, sp, cc = tables["bfcl_pools"], tables["pools"], tables["calling"]
    for p in pool_list:
        ev, m = p["eval_id"], p["model_id"]
        arm = (p["pool"] if p["pool"] not in POOL_OF_FAMILY else POOL_OF_FAMILY[p["pool"]])
        arm = {v: k for k, v in OBJECT_OF_ARM.items()}.get(arm, arm) + ("__model_balanced" if p["weighting"] == "model_balanced" else "")
        mk = "ALL" if m == "all" else MODEL_OF[m]
        src = sp[(sp["eval"] == EVALS[ev]) & (sp.model == mk) & (sp.arm == arm) & (sp.pool == "drawn")]
        if len(src) != 1:
            problems.append(f"{ev} {m} {p['pool']} ({p['weighting']}): {len(src)} rows in {FILES['pools']}")
            continue
        src = src.iloc[0]
        if set(src.runs.split()) != {_analysis_key(by_id[r]) for r in p["run_ids"]}:
            problems.append(f"{ev} {m} {p['pool']} ({p['weighting']}): the analysis pools other runs ({src.runs})")
        pt = pool_point(p, by_id)
        for q in ("lost", "gained", "delta_pp"):
            if not _close(pt[q], src[q]):
                problems.append(f"{ev} {m} {p['pool']}: {q} {pt[q]} from counts, {src[q]} in {FILES['pools']}")
        p["lost"], p["gained"] = _f(pt["lost"]), _f(pt["gained"])
        p["delta_pp"] = [_f(pt["delta_pp"]), _f(src.delta_lo), _f(src.delta_hi)]
        shares = {}
        if ev == "bfcl":
            row = bf[(bf.model == mk) & (bf.arm == arm) & (bf.pool == "drawn")]
            if len(row) != 1:
                problems.append(f"bfcl {m} {p['pool']}: {len(row)} rows in {FILES['bfcl_pools']}")
                continue
            row = row.iloc[0]
            if not _close(pt["delta_pp"], row.delta_pp) or not _close(pt["lost"], row.lost):
                problems.append(f"bfcl {m} {p['pool']}: lost / delta differ from {FILES['bfcl_pools']}")
            named = [(n, s) for n, s, _d in CLASSES["bfcl"] + EXTRAS["bfcl"]] + [(n, s) for n, s, _p in COMPOSITES["bfcl"]]
            for name, s in named:
                val = row[f"lost%{s}"]
                if not _close(pt["shares"][name], val):
                    problems.append(f"bfcl {m} {p['pool']}: {name} {pt['shares'][name]:.12g}% from counts, {val:.12g}% in {FILES['bfcl_pools']}")
                shares[name] = [_f(pt["shares"][name]), _f(row[f"lost%{s}_lo"]), _f(row[f"lost%{s}_hi"])]
        else:
            named = [(n, s) for n, s, _d in CLASSES[ev] + EXTRAS[ev]] + [(n, s) for n, s, _p in COMPOSITES[ev]]
            for name, s in named:
                val = src[f"lost%{s}"]
                if not _close(pt["shares"][name], val):
                    problems.append(f"{ev} {m} {p['pool']}: {name} {pt['shares'][name]:.12g}% from counts, {val:.12g}% in {FILES['pools']}")
                shares[name] = [_f(pt["shares"][name]), _f(src[f"lost%{s}_lo"]), _f(src[f"lost%{s}_hi"])]
        p["shares"] = shares
        if "calling" in pt:
            row = cc[(cc.model == mk) & (cc.arm == arm)]
            cal = {}
            for name, _d, _den in CALLING:
                for q in (name, "anchor_" + name):
                    val = pt["calling"][q]
                    if len(row) == 1:
                        if not _close(val, row.iloc[0][q]):
                            problems.append(f"bfcl {m} {p['pool']}: {q} {val} from counts, {row.iloc[0][q]} in {FILES['calling']}")
                        cal[q] = [_f(val), _f(row.iloc[0][q + "_lo"]), _f(row.iloc[0][q + "_hi"])]
                    else:
                        cal[q] = [_f(val), None, None]          # the analysis gave no interval for this pool (the prompted conditions)
            p["calling"] = cal


def _analysis_key(r: dict) -> str:
    """The run's key in the analysis's pool tables: the BFCL run key, or BigCodeBench's short run id."""
    return bfcl_key(r) if r["eval_id"] == "bfcl" else r["run_id"].split(":")[-1]


def attach_run_intervals(runs: list, tables: dict, problems: list) -> None:
    sp = tables["pools"]
    rr = sp[sp.pool == "run"].set_index(["eval", "runs"])
    for r in runs:
        key = (EVALS[r["eval_id"]], _analysis_key(r))
        if key not in rr.index:
            problems.append(f"{r['run_id']}: no per-run row in {FILES['pools']}")
            continue
        row = rr.loc[key]
        d = 100.0 * (r["gained"] - r["lost"]) / r["n"]
        for q, val in (("lost", r["lost"]), ("gained", r["gained"]), ("delta_pp", d)):
            if not _close(val, row[q]):
                problems.append(f"{r['run_id']}: {q} {val} from counts, {row[q]} in {FILES['pools']}")
        if r.get("drawn_delta_pp") is not None and not _close(d, r["drawn_delta_pp"]):
            problems.append(f"{r['run_id']}: delta {d} from counts, {r['drawn_delta_pp']} drawn by the paper notebook (manifest)")
        r["delta_pp"] = [_f(d), _f(row.delta_lo), _f(row.delta_hi)]


def contrasts(tables: dict, pool_list: list, problems: list) -> list:
    """The analysis's paired contrasts per model and family (control minus constitution; the difference object over the constitution
    and the control object), shared item bootstrap. Point values are checked against the pools' deltas."""
    delta = {(p["eval_id"], p["model_id"], p["pool"]): p["delta_pp"][0] for p in pool_list if p["weighting"] == "runs" and "delta_pp" in p}
    model_id = {v: k for k, v in MODEL_OF.items()}
    out = []
    for _, r in tables["premises"].iterrows():
        ev = {v: k for k, v in EVALS.items()}[r["eval"]]
        m = model_id[r.model]
        rec = dict(eval_id=ev, model_id=m, family=r.family)
        for role in ("control", "constitution", "difference"):
            want = delta.get((ev, m, f"{role}_{r.family}"))
            if not _close(want, r[f"{role}_pp"]):
                problems.append(f"contrast {ev} {m} {r.family}: {role} delta {r[f'{role}_pp']} in {FILES['premises']}, {want} from counts")
        for q in ("ctl_minus_con_pp", "ratio_diff_over_con", "ratio_diff_over_ctl", "diff_minus_con_pp"):
            base = q[:-3] if q.endswith("_pp") else q
            rec[q] = [_f(r[q]), _f(r[base + "_lo"]), _f(r[base + "_hi"])]
        out.append(rec)
    return out


def churn(tables: dict, unsteered: list, problems: list) -> list:
    """Unsteered against unsteered: per pair of one model's unsteered runs (every unsteered run in the manifest), the items the second
    loses and gains against the first, the run-to-run noise floor of a new-failure count."""
    key_of = {}
    for a in unsteered:
        k = f"{MODEL_OF[a['model_id']]}:unsteered:{_view_suffix(a['view'])}" if a["eval_id"] == "bfcl" else a["run_id"].split(":")[-1]
        key_of[(a["eval_id"], k)] = a["run_id"]
    out = []
    for _, r in tables["churn"].iterrows():
        ev = {v: k for k, v in EVALS.items()}[r["eval"]]
        ids = [key_of.get((ev, r.a)), key_of.get((ev, r.b))]
        if None in ids:
            problems.append(f"{FILES['churn']}: pair {r.a} / {r.b} names a run that is not an unsteered run of the manifest")
            continue
        out.append(dict(eval_id=ev, model_id={v: k for k, v in MODEL_OF.items()}[r.model], a=ids[0], b=ids[1],
                        b_lost=_count(r.flips_b_lost), b_gained=_count(r.flips_b_gained), n=int(r.n_items)))
    return out


# ---- the binding to the paper snapshot --------------------------------------------------------------------------------------------
def snapshot_dir() -> Path:
    cur = (SNAPSHOT_ROOT / "CURRENT").read_text().strip()
    return SNAPSHOT_ROOT / "snapshots" / cur


def snapshot_check(runs: list, anchors: list, problems: list) -> dict:
    """Recompute k, n, new failures and gains of every run from the committed snapshot's item results, and read each run's object id
    from its run table; every value must equal the sidecar's."""
    import pyarrow.parquet as pq
    snap = snapshot_dir()
    ids = sorted({r["run_id"] for r in runs} | {a["run_id"] for a in anchors})
    ir = pq.read_table(snap / "item_result.parquet", columns=["run_id", "item_id", "metric_id", "trial_index", "value", "withdrawn_by_revision_id"],
                       filters=[("run_id", "in", ids)]).to_pandas()
    ir = ir[ir.withdrawn_by_revision_id.isna() & (ir.metric_id == "acc")]
    if ir.duplicated(["run_id", "item_id", "trial_index"]).any():
        problems.append("snapshot item_result: a run holds two rows for one item")
    vec = {rid: g.set_index("item_id").value for rid, g in ir.groupby("run_id")}
    rt = pq.read_table(snap / "run.parquet", columns=["run_id", "object_id", "model_id", "eval_id", "item_set_id"],
                       filters=[("run_id", "in", ids)]).to_pandas().set_index("run_id")
    n_ok = 0
    for r in runs:
        s, a = vec.get(r["run_id"]), vec.get(r["anchor_run_id"])
        if s is None or a is None:
            problems.append(f"{r['run_id']}: no item results in the snapshot for the run or its anchor")
            continue
        a = a.reindex(s.index)
        if a.isna().any():
            problems.append(f"{r['run_id']}: the snapshot's anchor items differ from the run's")
            continue
        got = dict(k=int(s.sum()), n=len(s), lost=int(((1 - s) * a).sum()), gained=int((s * (1 - a)).sum()))
        bad = {q: (got[q], r[q]) for q in got if got[q] != r[q]}
        if r["run_id"] not in rt.index or rt.loc[r["run_id"], "object_id"] != r["object_id"]:
            bad["object_id"] = (rt.loc[r["run_id"], "object_id"] if r["run_id"] in rt.index else None, r["object_id"])
        if bad:
            problems.append(f"{r['run_id']}: the snapshot gives {bad} (snapshot, sidecar)")
        else:
            n_ok += 1
    return {"snapshot": snap.name, "runs_checked": len(runs), "runs_matching": n_ok,
            "fields": ["k", "n", "lost", "gained", "object_id"], "source": "paper/snapshot item_result.parquet (metric acc) and run.parquet"}


# ---- assembly ---------------------------------------------------------------------------------------------------------------------
def build(src: Path) -> dict:
    missing = [rel for rel in list(FILES.values()) + list(RULE_FILES.values()) if not (src / rel).exists()]
    if missing:
        raise BuildError(f"source {display(src)} lacks {missing}; point --source or {SOURCE_ENV} at the error analysis's work directory")
    manifest = json.loads((src / FILES["manifest"]).read_text())
    drawn, anchors = scope(manifest)
    tables = {k: pd.read_csv(src / FILES[k]) for k in ("bfcl_pools", "pools", "calling", "premises", "churn")}
    bf_runs = [r for r in drawn if r["eval_id"] == "bfcl"]
    bc_runs = [r for r in drawn if r["eval_id"] == "bigcodebench"]
    anc_bf = [a for a in anchors if a["eval_id"] == "bfcl"]
    anc_bc = [a for a in anchors if a["eval_id"] == "bigcodebench"]
    arr_bf, items, cats = load_bfcl(src, bf_runs + anc_bf)
    arr_bc = load_bcb(src, bc_runs + anc_bc)
    arrays = {**arr_bf, **arr_bc}
    problems = []
    runs = []
    for r in sorted(drawn, key=lambda r: (list(EVALS).index(r["eval_id"]), MODEL_ORDER.index(r["model_id"]),
                                           (FITTED + PROMPTED).index(OBJECT_OF_ARM.get(r["arm"], r["arm"])), r.get("seed") or "", r["view"])):
        c = run_counts(r, arrays[r["run_id"]], arrays[r["_anchor"]], r["eval_id"])
        if c["k"] != int(r["k"]) or c["n"] != int(r["n"]):
            problems.append(f"{r['run_id']}: k/n {c['k']}/{c['n']} from the class table, {r['k']}/{r['n']} in the manifest")
        a = next(x for x in anchors if x["run_id"] == r["_anchor"])
        if c["anchor_k"] != int(a["k"]):
            problems.append(f"{r['run_id']}: anchor k {c['anchor_k']} from the class table, {a['k']} in the manifest")
        rec = dict(run_id=r["run_id"], object_id=r["object_id"], model_id=r["model_id"], eval_id=r["eval_id"],
                   object_label=OBJECT_OF_ARM.get(r["arm"], r["arm"]), seed=r.get("seed") or "", view=r["view"],
                   transfer_kind=r.get("transfer_kind"), item_set_id=r["item_set_id"], token_cap_tokens=_f(r.get("token_cap_tokens")),
                   anchor_run_id=r["_anchor"], drawn_delta_pp=_f(r.get("delta_pp")))
        rec.update(c)
        runs.append(rec)
    by_id = {r["run_id"]: r for r in runs}
    attach_run_intervals(runs, tables, problems)
    for r in runs:
        r.pop("drawn_delta_pp", None)
    pool_list = pools(runs)
    attach_intervals(pool_list, by_id, tables, problems)
    con = contrasts(tables, pool_list, problems)
    anc = [dict(run_id=a["run_id"], model_id=a["model_id"], eval_id=a["eval_id"], view=a["view"], k=int(a["k"]), n=int(a["n"]),
                item_set_id=a["item_set_id"]) for a in anchors]
    ch = churn(tables, [x for x in manifest["runs"] if x["arm"] == "unsteered" and x["eval_id"] in EVALS], problems)
    snap = snapshot_check(runs, anchors, problems)
    n_items = {"bfcl": len(items), "bigcodebench": 1140}
    n_rel = {r["calling"]["n_relevant"] for r in runs if r["eval_id"] == "bfcl"}
    if len(n_rel) != 1:
        problems.append(f"BFCL runs disagree on how many items require a call: {sorted(n_rel)}")
    if problems:
        raise BuildError(f"{len(problems)} problem(s), nothing written:\n  " + "\n  ".join(problems[:40]))
    n_rel = n_rel.pop()
    return {
        "schema": SCHEMA,
        "judge": {"kind": "deterministic", "judge_id": None, "judge_model": None,
                  "text": "No LLM judge: BFCL is scored by its abstract-syntax-tree checker and BigCodeBench by its unit tests (the "
                          "registry's per-item acc); each new failure is classed by deterministic rules over the stored response, the "
                          "parsed tool calls, the extracted program and the test output."},
        "provenance": {
            "what": "Counts of new failures by class for every BFCL and BigCodeBench run the capability figures draw (Figure 6 and the "
                    "all-model capability figure), with the error analysis's item-bootstrap intervals for each pool.",
            "source": "the BFCL and BigCodeBench error analysis's work directory, outside the repository (its class tables hold "
                      "response text); every path below is relative to it",
            "files": {k: {"path": rel, "sha256": sha256(src / rel)} for k, rel in FILES.items()},
            "rule_files": {k: {"path": rel, "sha256": sha256(src / rel)} for k, rel in RULE_FILES.items()},
            "builder": "paper/notebooks/data/build_appendix_caperrors.py",
            "registry_head": int(manifest["meta"]["primary_source"]["registry_head"]),
            "snapshot": Path(manifest["meta"]["primary_source"]["path"]).name,
            "bootstrap": {"B": 4000, "seed": 20261111, "kind": "item-cluster multinomial resample over items (every run of a pool moves "
                          "together on an item), percentile 95% intervals; seed-to-seed variance is not in them"},
            "new_failure": "an item the run's seed-matched unsteered run (anchor_run_id) answered correctly and the run did not; a gain is "
                           "the reverse. A pool sums its runs' counts.",
            "snapshot_check": snap,
        },
        "evals": {ev: {"n_items": n_items[ev], "classes": [dict(name=n, source_label=s, definition=d) for n, s, d in CLASSES[ev]],
                       "extras": [dict(name=n, source_label=s, definition=d) for n, s, d in EXTRAS[ev]],
                       "composites": {k: v for k, v in COMPOSITE_PARTS[ev].items()}}
                  for ev in EVALS},
        "bfcl_item_kinds": {"n_relevant": int(n_rel), "n_irrelevance": int(n_items["bfcl"] - n_rel),
                            "categories": {str(k): int(v) for k, v in pd.Series(cats).value_counts().sort_index().items()}},
        "anchors": anc,
        "runs": runs,
        "pools": pool_list,
        "contrasts": con,
        "anchor_churn": ch,
    }


def _round(x):
    """Floats to 12 significant digits so a rebuild is byte-stable across platforms."""
    if isinstance(x, float):
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def dumps(side: dict) -> str:
    return json.dumps(_round(side), indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=os.environ.get(SOURCE_ENV), help=f"the error analysis's work directory (or {SOURCE_ENV})")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with --out; write nothing")
    a = ap.parse_args(argv)
    if not a.source:
        print(f"build_appendix_caperrors: REFUSED: name the error analysis's work directory with --source or {SOURCE_ENV}", file=sys.stderr)
        return 2
    src = Path(os.path.expanduser(a.source))
    try:
        side = build(src)
    except BuildError as e:
        print(f"build_appendix_caperrors: REFUSED: {e}", file=sys.stderr)
        return 2
    text = dumps(side)
    if a.check:
        old = json.loads(Path(a.out).read_text())
        new = json.loads(text)
        if old == new:
            print(f"{display(a.out)} matches a fresh build from {display(src)}")
            return 0
        diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        print(f"{display(a.out)} DIFFERS from a fresh build in {diff}", file=sys.stderr)
        return 1
    Path(a.out).write_text(text)
    nb = sum(r["eval_id"] == "bfcl" for r in side["runs"])
    print(f"wrote {display(a.out)}: {len(side['runs'])} runs ({nb} BFCL, {len(side['runs']) - nb} BigCodeBench), {len(side['anchors'])} anchors, "
          f"{len(side['pools'])} pools, {len(side['contrasts'])} contrasts; snapshot check {side['provenance']['snapshot_check']['runs_matching']}"
          f"/{side['provenance']['snapshot_check']['runs_checked']} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
