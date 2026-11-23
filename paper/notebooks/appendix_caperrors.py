"""appendix_caperrors: the error analysis of the BFCL and BigCodeBench losses (appendix `app:caperrors`).

    import appendix_caperrors as CE
    side = CE.load_caperrors()                                   # the committed sidecar, validated (refuses by name)
    check = CE.item_check(side, R.query(CE.item_sql(side), "caperrors_items"), R.query(CE.object_sql(side), "caperrors_objects"))
    CE.assert_bound(check)                                       # every run's k, n, new failures, gains and object id, from the source read
    CE.qwen_figure(side, "figA_qwen_capability_errors")          # Qwen3.5-4B, BFCL beside BigCodeBench
    CE.models_figure(side, "figA_models_capability_errors")      # every model, one row of panels each
    print(CE.table_caperrors(side)); print(CE.table_caperrors_bcb(side))     # tab:caperrors, tab:caperrors-bcb

WHAT IS COUNTED. A NEW FAILURE is an item the run's seed-matched unsteered run answered correctly and the run did not (a gain is the
reverse). Each new failure takes one class, assigned by deterministic rules over the stored response, the parsed tool calls, the extracted
program and the test output; BFCL is scored by its abstract-syntax-tree checker and BigCodeBench by its unit tests. No LLM judge enters
any number here (the sidecar's `judge` stamp says so and `load_caperrors` refuses any other stamp). A pool (Qwen3.5-4B's three seeds of one
object, a family of objects, every model) sums its runs' counts: a class share is the pooled class count over the pooled new failures, and
the change in pass rate is the pooled gains minus new failures over the pooled items. The intervals are the error analysis's 95% item
bootstrap (B = 4000; every run of a pool moves together on an item), which leaves out seed-to-seed variance.

THE SIDECAR, `data/appendix_caperrors.json`, holds per run (pinned by run id, object id and anchor run id) the counts of new failures by
class and a few further counts over them, the gains by class and the BFCL calling counts; per pool its runs and the analysis's intervals.
`data/build_appendix_caperrors.py` writes it from the analysis's per-item class tables (which hold response text and are not committed)
and checks every point against the analysis's own tables and the committed snapshot's per-item results before writing. This module
recomputes every point value from the run counts (`pool_point`) and refuses a sidecar whose stored points disagree; the notebook then
recomputes each run's correct count, new failures and gains from the per-item results of the source it reads, and the run's object id
from its run table (`item_check`), so the figures cannot drift from the registry's runs without the render stopping.

THE FIGURES are compositions (non-comparison figures): one stacked bar per object or pool, the classes as shares of its new failures in a
fixed order, the share printed in any segment of at least 18%, and beside the bar the headline shares, the change in pass rate (pp) and
the number of new failures. The style is the paper's (`R.style()` fonts and sizes, the ink colours, `R.LABEL` and `R.MODEL_LABEL` names,
the legend on top, `appendix_common.save_figure`'s png / pdf / svg); the six class colours are the analysis figure's validated categorical
palette, not object colours, because a class is not an object.
"""
from __future__ import annotations

import json
import math
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import registry_nb as R
import appendix_common as AC

SIDECAR = os.path.join(R.HERE, "data", "appendix_caperrors.json")
SCHEMA = "appendix-caperrors-v1"
EVALS = ("bfcl", "bigcodebench")
EVAL_LABEL = {"bfcl": "BFCL", "bigcodebench": "BigCodeBench"}
MODELS = ("qwen35_4b_it", "gemma4e4b_it", "nemotron3nano4b", "llama8b_it")          # the tables' and the models figure's order
FITTED = ("control_relu", "constitution_relu", "difference_relu", "control_lora", "constitution_lora", "difference_lora")
FAMILY_LABEL = {"all_relu": "All ReLU steers", "all_lora": "All LoRAs", "all_fitted": "All fitted objects"}
# THE ROWS of a composition panel, top to bottom; None is a gap between groups. The Refusal steer has capability runs on Qwen3.5-4B only.
ROWS = ("control_relu", "constitution_relu", "difference_relu", "all_relu", None,
        "control_lora", "constitution_lora", "difference_lora", "all_lora", None,
        "safety_si", "constitution_si", "arditi_allpos")
GAP_SLOT = 0.6                     # a gap row's height, in bar rows
BAR_HEIGHT = 0.62                  # a bar's height, in bar rows
LABEL_MIN = 18.0                   # a segment this wide (% of the bar) or wider prints its share
# THE CLASSES in stacking order (the sidecar's order, which `load_caperrors` holds the file to), each with its colour and legend text. The
# colours are the analysis figure's six-slot categorical palette, validated in this order (adjacent pairs) with the dataviz skill's
# validate_palette.js on a white surface: lightness band, chroma floor, CVD separation (worst adjacent dE 9.1) and normal-vision floor
# pass; three fills sit under 3:1 against white, so wide segments carry their share and the tables give every number. The slots carry a
# matched meaning across the two evaluations: prose instead of a call or code; a declination or refusal; a malformed call or program; a
# run to the token cap; wrong content; a call on an irrelevance item or an undecidable self-test.
CLASS_STYLE = {
    "bfcl": (("no_call_other", "#2a78d6", "No call: answers, asks, or says it has no access"),
             ("no_call_declination", "#eb6834", "No call: explicit declination ('I can't do that', 'I'm not going to call ...')"),
             ("malformed_call", "#1baf7a", "Format: malformed call (attempted, not parsed intact)"),
             ("call_then_cap", "#eda100", "Format: call parsed, then ran to the 1,024-token cap"),
             ("wrong_call", "#e87ba4", "Wrong call (intact call, wrong function, arguments or count)"),
             ("irrelevance_call", "#008300", "Called on an irrelevance item (should not call)")),
    "bigcodebench": (("no_code_prose", "#2a78d6", "Format: no code, prose instead (not a refusal)"),
                     ("refusal_or_objection", "#eb6834", "Refusal or objection to the task (keyword rule; a lower bound for the control ReLU)"),
                     ("malformed_code", "#1baf7a", "Format: malformed or mis-extracted code"),
                     ("cap_no_program", "#eda100", "Format: ran to the 4,096-token cap, no program"),
                     ("wrong_code", "#e87ba4", "Wrong code (logic, API or syntax error)"),
                     ("undecidable", "#008300", "Only the answer's own appended self-test fails (undecidable)")),
}
ON_FILL = {"#eda100": R.INK}       # share text on a fill: white, except ink on the light yellow
TOL = 1e-6                         # the sidecar stores 12 significant digits

# THE TABLES' TEXT: Table 6's caption and block labels as the paper sets them, passed by default so the printed table can be pasted (a
# caller passes its own); NOTE_BFCL, the fuller reading of the same table the notebook prints under it; and the caption of the
# BigCodeBench sibling, which the notebook alone prints.
TABLE_BLOCK = {"qwen35_4b_it": "Qwen3.5-4B (seeds 1--3 pooled)", "gemma4e4b_it": "Gemma-4-E4B (one fit per arm)",
               "nemotron3nano4b": "Nemotron 3 Nano 4B (one fit per arm)", "llama8b_it": "Llama-3.1-8B-Instruct (one fit per arm)"}
TABLE_ROW = {"control_relu": "Control ReLU", "constitution_relu": "Constitution ReLU", "difference_relu": "Difference ReLU",
             "control_lora": "Control LoRA", "constitution_lora": "Constitution LoRA", "difference_lora": "Difference LoRA"}
CAPTION_BFCL = (
    "BFCL pass rate and new failures by class. First row of each block: unsteered pass rate (\\%). Other rows: change against the "
    "seed-matched unsteered run (pp, 95\\% item-bootstrap interval), the number of new failures, and their classes as a percentage, with "
    "explicit declinations in parentheses. An irrelevance call is a call on an item that needs none.")
NOTE_BFCL = (
    "How to read Table 6. A new failure is an item the seed-matched unsteered run answered correctly and the steered run did not. The "
    "first row of each block is the unsteered pass rate, averaged over seeds where there are seed replicates (" + R.SEED_AVERAGE_TIES + "); "
    "the change (pp) is the steered run's against its own seed-matched unsteered run (positive = better), so first row + change "
    "approximates the steered pass rate, exactly where that run equals the first row. Brackets are 95% item-bootstrap intervals that "
    "leave out seed-to-seed variance. The five class columns are percentages of the new failures and sum to 100 up to rounding. 'No call' "
    "is a response with no tool call at all, and the parenthesis gives the share of new failures whose wording is an explicit declination "
    "(a keyword rule). 'Call, then cap' is a parsed call followed by text that ran to the 1,024-token cap; 'irrelevance call' is a call on "
    "an item where the correct answer is to call nothing. Shares over fewer than 100 new failures (Gemma-4-E4B's LoRAs) are imprecise. "
    "Llama-3.1-8B-Instruct has no BFCL run.")
CAPTION_BCB = (
    "What the new BigCodeBench failures are, laid out as in Table~\\ref{tab:caperrors}. Format failures are malformed or mis-extracted "
    "programs, prose with no code, and responses that reached the 4,096-token cap with no program. ``Refusal or objection'' is a keyword "
    "rule over the text before the code: it is an upper bound on content-caused loss for the constitution objects (some code after an "
    "objection would have failed anyway) and a lower bound for the control ReLU steer (3--7\\% by wording-neutral rules). ``Undecidable'' "
    "answers fail only a self-test they appended themselves. Rows sum to 100 up to rounding.")


def _stack(*lines) -> str:
    """A table header cell of stacked lines, as the manuscript's tables set them."""
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


_PASS_HEAD = _stack("Pass rate (\\%)", "or $\\Delta$ (pp)")
_NEW_HEAD = _stack("New", "failures")


# ---- the sidecar --------------------------------------------------------------------------------------------------------------------
def _close(a, b, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


def pool_weights(run_ids: list, model_of: dict, weighting: str) -> dict:
    """{run id: weight}: 1 each for a run-weighted pool; for a model-balanced one 1 / (that model's runs in the pool), renormalised so
    the weights sum to the pool's run count (each model then counts equally)."""
    if weighting == "runs":
        return {r: 1.0 for r in run_ids}
    if weighting != "model_balanced":
        raise ValueError(f"weighting {weighting!r}: 'runs' or 'model_balanced'")
    per = {}
    for r in run_ids:
        per.setdefault(model_of[r], []).append(r)
    tot = sum(1.0 / len(per[model_of[r]]) for r in run_ids)
    return {r: (1.0 / len(per[model_of[r]])) * len(run_ids) / tot for r in run_ids}


def pool_point(side: dict, pool: dict) -> dict:
    """The pool's point values recomputed from its runs' counts: lost and gained (new failures and gains, weighted), the change in pass
    rate (pp), every class / extra / composite as % of the new failures, and (BFCL, run-weighted) the calling rates in % of the pooled
    items of each kind, the run's own and its anchors'."""
    by_id = {r["run_id"]: r for r in side["runs"]}
    ev = pool["eval_id"]
    rs = [by_id[r] for r in pool["run_ids"]]
    w = pool_weights(pool["run_ids"], {r["run_id"]: r["model_id"] for r in rs}, pool["weighting"])
    lost = sum(w[r["run_id"]] * r["lost"] for r in rs)
    gained = sum(w[r["run_id"]] * r["gained"] for r in rs)
    out = {"lost": lost, "gained": gained, "delta_pp": 100.0 * (gained - lost) / (rs[0]["n"] * sum(w.values()))}
    share = lambda c: 100.0 * c / lost if lost else float("nan")  # noqa: E731
    shares = {c["name"]: share(sum(w[r["run_id"]] * r["lost_by_class"][c["name"]] for r in rs)) for c in side["evals"][ev]["classes"]}
    shares.update({c["name"]: share(sum(w[r["run_id"]] * r["lost_extra"][c["name"]] for r in rs)) for c in side["evals"][ev]["extras"]})
    shares.update({k: sum(shares[c] for c in parts) for k, parts in side["evals"][ev]["composites"].items()})
    out["shares"] = shares
    if ev == "bfcl" and pool["weighting"] == "runs":
        cal = {}
        for name, den in (("relevant_no_call", "n_relevant"), ("irrelevance_called", "n_irrelevance")):
            d = sum(r["calling"][den] for r in rs)
            cal[name] = 100.0 * sum(r["calling"][name] for r in rs) / d
            cal["anchor_" + name] = 100.0 * sum(r["calling"]["anchor_" + name] for r in rs) / d
        out["calling"] = cal
    return out


RUN_FIELDS = ("run_id", "object_id", "model_id", "eval_id", "object_label", "seed", "view", "anchor_run_id", "k", "n", "anchor_k", "lost",
              "gained", "lost_by_class", "lost_extra", "gained_by_class", "delta_pp")


def caperrors_problems(side: dict) -> list:
    """Every way the sidecar fails its contract, one phrase each (empty when it holds): the schema; the judge stamp (deterministic, no
    judge); the provenance block (the source files with their sha256, the registry head and snapshot, the bootstrap, a snapshot check that
    matched every run); the class order the figures draw; each run's fields, its classes summing to its new failures and its gains, its
    delta equal to its counts, its anchor among the anchors with the same k; each pool's runs (present, of its eval and model, unique) and
    every stored point equal to the one `pool_point` recomputes; every interval holding its point; the contrasts' points equal to the
    pools' deltas."""
    p = []
    if side.get("schema") != SCHEMA:
        return [f"schema {side.get('schema')!r}, expected {SCHEMA!r}"]
    j = side.get("judge") or {}
    if j.get("kind") != "deterministic" or j.get("judge_id") is not None or j.get("judge_model") is not None:
        p.append(f"judge stamp {j!r}: this analysis is judge-free (kind 'deterministic', no judge id or model)")
    prov = side.get("provenance") or {}
    for k in ("source", "files", "rule_files", "registry_head", "snapshot", "bootstrap", "snapshot_check", "builder"):
        if k not in prov:
            p.append(f"provenance lacks {k!r}")
    for k, f in {**prov.get("files", {}), **prov.get("rule_files", {})}.items():
        if not (isinstance(f, dict) and f.get("path") and isinstance(f.get("sha256"), str) and len(f["sha256"]) == 64):
            p.append(f"provenance file {k!r} lacks a path and a sha256")
    sc = prov.get("snapshot_check") or {}
    if not (sc.get("runs_checked") == sc.get("runs_matching") == len(side.get("runs", []))):
        p.append(f"snapshot check matched {sc.get('runs_matching')} of {sc.get('runs_checked')} runs; the sidecar holds {len(side.get('runs', []))}")
    for ev in EVALS:
        want = [c for c, _col, _lab in CLASS_STYLE[ev]]
        got = [c.get("name") for c in side.get("evals", {}).get(ev, {}).get("classes", [])]
        if got != want:
            p.append(f"{ev} classes {got}, the figures draw {want}")
    if p:
        return p
    anchors = {a["run_id"]: a for a in side.get("anchors", [])}
    runs = side.get("runs", [])
    ids = [r.get("run_id") for r in runs]
    if len(set(ids)) != len(ids):
        p.append("a run id repeats")
    classes = {ev: [c["name"] for c in side["evals"][ev]["classes"]] for ev in EVALS}
    extras = {ev: [c["name"] for c in side["evals"][ev]["extras"]] for ev in EVALS}
    for r in runs:
        miss = [f for f in RUN_FIELDS if f not in r]
        if miss:
            p.append(f"run {r.get('run_id')} lacks {miss}")
            continue
        rid, ev = r["run_id"], r["eval_id"]
        if ev not in EVALS or r["model_id"] not in MODELS:
            p.append(f"run {rid}: eval {ev!r} / model {r['model_id']!r} outside the figures")
            continue
        if sorted(r["lost_by_class"]) != sorted(classes[ev]) or sorted(r["gained_by_class"]) != sorted(classes[ev]):
            p.append(f"run {rid}: class keys differ from the eval's classes")
            continue
        if sorted(r["lost_extra"]) != sorted(extras[ev]):
            p.append(f"run {rid}: extra keys differ from the eval's extras")
        if not (0 <= r["k"] <= r["n"]) or r["n"] != side["evals"][ev]["n_items"]:
            p.append(f"run {rid}: k {r['k']} of n {r['n']}, the eval has {side['evals'][ev]['n_items']} items")
        if sum(r["lost_by_class"].values()) != r["lost"]:
            p.append(f"run {rid}: classes sum to {sum(r['lost_by_class'].values())}, new failures {r['lost']}")
        if sum(r["gained_by_class"].values()) != r["gained"]:
            p.append(f"run {rid}: gained classes sum to {sum(r['gained_by_class'].values())}, gains {r['gained']}")
        if any(v < 0 for v in list(r["lost_by_class"].values()) + list(r["lost_extra"].values())):
            p.append(f"run {rid}: a negative count")
        if any(v > r["lost"] for v in r["lost_extra"].values()):
            p.append(f"run {rid}: an extra count exceeds the new failures")
        d = 100.0 * (r["gained"] - r["lost"]) / r["n"]
        if not _close(r["delta_pp"][0], d):
            p.append(f"run {rid}: delta {r['delta_pp'][0]} stored, {d} from its counts")
        if r["k"] - r["anchor_k"] != r["gained"] - r["lost"]:
            p.append(f"run {rid}: k - anchor k = {r['k'] - r['anchor_k']}, gains - new failures = {r['gained'] - r['lost']}")
        a = anchors.get(r["anchor_run_id"])
        if a is None or a["model_id"] != r["model_id"] or a["eval_id"] != ev or a["k"] != r["anchor_k"] or a["n"] != r["n"]:
            p.append(f"run {rid}: anchor {r['anchor_run_id']} is not an anchor of its model and eval with k {r['anchor_k']}")
        if ev == "bfcl":
            c = r.get("calling") or {}
            if c.get("n_relevant", -1) + c.get("n_irrelevance", -1) != r["n"]:
                p.append(f"run {rid}: calling counts do not cover its items")
    by_id = {r["run_id"]: r for r in runs}
    for q in side.get("pools", []):
        tag = f"pool {q.get('eval_id')} {q.get('model_id')} {q.get('pool')} ({q.get('weighting')})"
        rids = q.get("run_ids") or []
        if not rids or len(set(rids)) != len(rids) or any(r not in by_id for r in rids):
            p.append(f"{tag}: run ids missing, repeated or unknown")
            continue
        if any(by_id[r]["eval_id"] != q["eval_id"] or (q["model_id"] != "all" and by_id[r]["model_id"] != q["model_id"]) for r in rids):
            p.append(f"{tag}: holds a run of another eval or model")
            continue
        pt = pool_point(side, q)
        for k in ("lost", "gained"):
            if not _close(q.get(k), pt[k]):
                p.append(f"{tag}: {k} {q.get(k)} stored, {pt[k]} from its runs")
        if not _close((q.get("delta_pp") or [None])[0], pt["delta_pp"]):
            p.append(f"{tag}: delta {(q.get('delta_pp') or [None])[0]} stored, {pt['delta_pp']} from its runs")
        for k, v in (q.get("shares") or {}).items():
            if k not in pt["shares"] or not _close(v[0], pt["shares"][k]):
                p.append(f"{tag}: share {k} {v[0]} stored, {pt['shares'].get(k)} from its runs")
        if sorted(q.get("shares") or {}) != sorted(pt["shares"]):
            p.append(f"{tag}: shares {sorted(q.get('shares') or {})}, expected {sorted(pt['shares'])}")
        for k, v in (q.get("calling") or {}).items():
            if not _close(v[0], (pt.get("calling") or {}).get(k)):
                p.append(f"{tag}: calling {k} {v[0]} stored, {(pt.get('calling') or {}).get(k)} from its runs")
    for tag, v in _intervals(side):
        pt, lo, hi = v
        if pt is not None and lo is not None and hi is not None and not (lo - TOL <= pt <= hi + TOL):
            p.append(f"{tag}: interval [{lo}, {hi}] does not hold {pt}")
        if (lo is None) != (hi is None):
            p.append(f"{tag}: one interval bound only")
    delta = {(q["eval_id"], q["model_id"], q["pool"]): q["delta_pp"][0] for q in side.get("pools", []) if q.get("weighting") == "runs"}
    for c in side.get("contrasts", []):
        ctl, con = (delta.get((c["eval_id"], c["model_id"], f"{role}_{c['family']}")) for role in ("control", "constitution"))
        dif = delta.get((c["eval_id"], c["model_id"], f"difference_{c['family']}"))
        if None in (ctl, con, dif):
            p.append(f"contrast {c['eval_id']} {c['model_id']} {c['family']}: a pool it contrasts is missing")
            continue
        if not _close(c["ctl_minus_con_pp"][0], ctl - con):
            p.append(f"contrast {c['eval_id']} {c['model_id']} {c['family']}: control - constitution {c['ctl_minus_con_pp'][0]}, pools give {ctl - con}")
        if con and c["ratio_diff_over_con"][0] is not None and not _close(c["ratio_diff_over_con"][0], dif / con):
            p.append(f"contrast {c['eval_id']} {c['model_id']} {c['family']}: difference / constitution {c['ratio_diff_over_con'][0]}, pools give {dif / con}")
    return p


def _intervals(side: dict):
    for q in side.get("pools", []):
        tag = f"pool {q['eval_id']} {q['model_id']} {q['pool']} ({q['weighting']})"
        yield f"{tag} delta", q.get("delta_pp") or [None, None, None]
        for k, v in (q.get("shares") or {}).items():
            yield f"{tag} share {k}", v
        for k, v in (q.get("calling") or {}).items():
            yield f"{tag} calling {k}", v
    for r in side.get("runs", []):
        yield f"run {r['run_id']} delta", r.get("delta_pp") or [None, None, None]
    for c in side.get("contrasts", []):
        for k in ("ctl_minus_con_pp", "ratio_diff_over_con", "ratio_diff_over_ctl", "diff_minus_con_pp"):
            yield f"contrast {c['eval_id']} {c['model_id']} {c['family']} {k}", c.get(k) or [None, None, None]


def load_caperrors(path: str = SIDECAR) -> dict:
    """The committed sidecar, after `caperrors_problems` finds nothing; otherwise a ValueError naming every problem (nothing is drawn
    from a sidecar that breaks its contract). Regenerate it with `data/build_appendix_caperrors.py`."""
    with open(path) as f:
        side = json.load(f)
    probs = caperrors_problems(side)
    if probs:
        raise ValueError(f"{R.display_path(path)}: {len(probs)} problem(s), nothing drawn — regenerate it with "
                         f"paper/notebooks/data/build_appendix_caperrors.py:\n  " + "\n  ".join(probs[:30]))
    return side


def pool(side: dict, eval_id: str, model_id: str, name: str, weighting: str = "runs") -> dict | None:
    for q in side["pools"]:
        if (q["eval_id"], q["model_id"], q["pool"], q["weighting"]) == (eval_id, model_id, name, weighting):
            return q
    return None


# ---- the binding to the registry ----------------------------------------------------------------------------------------------------
def _ids(side: dict) -> list:
    return sorted({r["run_id"] for r in side["runs"]} | {a["run_id"] for a in side["anchors"]})


def _in(ids) -> str:
    return ", ".join("'" + i.replace("'", "''") + "'" for i in ids)


def item_sql(side: dict) -> str:
    """The per-item results of every run and anchor in the sidecar (BFCL and BigCodeBench score one `acc` row per item)."""
    return ("SELECT run_id, item_id, trial_index, value FROM item_result WHERE metric_id = 'acc' AND withdrawn_by_revision_id IS NULL "
            f"AND run_id IN ({_in(_ids(side))})")


def object_sql(side: dict) -> str:
    return f"SELECT run_id, object_id, model_id, eval_id FROM run WHERE run_id IN ({_in(_ids(side))})"


def item_check(side: dict, items: pd.DataFrame, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per sidecar run: k, n, new failures and gains recomputed from the per-item results `items` (run_id, item_id, value; one
    row per item) against the run's anchor, the object id from `runs` (run_id, object_id) when given, beside the sidecar's, and `ok`
    (every field equal) with `why` naming what differs."""
    it = items.copy()
    if it.duplicated(["run_id", "item_id"]).any():
        raise ValueError("item results hold two rows for one (run, item): pass one row per item")
    vec = {rid: g.set_index("item_id").value.astype(float) for rid, g in it.groupby("run_id")}
    obj = runs.set_index("run_id").object_id.to_dict() if runs is not None and len(runs) else {}
    out = []
    for r in side["runs"]:
        rec = dict(run_id=r["run_id"], model_id=r["model_id"], eval_id=r["eval_id"], object_label=r["object_label"], seed=r["seed"],
                   k=np.nan, n=np.nan, lost=np.nan, gained=np.nan, object_id=None, ok=False, why="")
        s, a = vec.get(r["run_id"]), vec.get(r["anchor_run_id"])
        if s is None or a is None:
            rec["why"] = "no item results for the run" if s is None else "no item results for its anchor"
            out.append(rec)
            continue
        if set(a.index) != set(s.index):
            rec["why"] = "the run and its anchor hold different items"
            out.append(rec)
            continue
        a = a.reindex(s.index)
        rec.update(k=float(s.sum()), n=float(len(s)), lost=float(((1 - s) * a).sum()), gained=float((s * (1 - a)).sum()))
        bad = [f"{q} {rec[q]:g} (sidecar {r[q]})" for q in ("k", "n", "lost", "gained") if rec[q] != r[q]]
        if runs is not None:
            rec["object_id"] = obj.get(r["run_id"])
            if rec["object_id"] != r["object_id"]:
                bad.append(f"object {str(rec['object_id'])[:12]} (sidecar {r['object_id'][:12]})")
        rec["ok"], rec["why"] = not bad, "; ".join(bad)
        out.append(rec)
    return pd.DataFrame(out)


def assert_bound(check: pd.DataFrame) -> None:
    """Refuse (ValueError) when any run's recomputed k, n, new failures, gains or object id differ from the sidecar's: the sidecar no
    longer describes the runs the source holds, so nothing is drawn from it."""
    bad = check[~check.ok]
    if len(bad):
        raise ValueError(f"{len(bad)} of {len(check)} capability error runs disagree with the source's per-item results — rebuild the "
                         "sidecar (paper/notebooks/data/build_appendix_caperrors.py): "
                         + "; ".join(f"{r.run_id} {r.why}" for r in bad.head(8).itertuples()))


def binding_line(side: dict, check: pd.DataFrame) -> str:
    prov = side["provenance"]
    return (f"capability error sidecar bound to the source: {int(check.ok.sum())} of {len(check)} runs' correct counts, new failures, gains "
            f"and object ids recomputed from its per-item results and run table match · {sum(r['eval_id'] == 'bfcl' for r in side['runs'])} "
            f"BFCL and {sum(r['eval_id'] == 'bigcodebench' for r in side['runs'])} BigCodeBench runs, {len(side['anchors'])} unsteered anchors · "
            f"built at registry head {prov['registry_head']} · judge: none (deterministic scoring and classes) · "
            f"intervals: {prov['bootstrap']['B']} item resamples, seed {prov['bootstrap']['seed']}")


# ---- the composition --------------------------------------------------------------------------------------------------------------
def headline(eval_id: str, shares: dict) -> str:
    """The text beside a bar: BFCL's no-call share and format share (malformed calls and calls run to the cap), BigCodeBench's format
    share and refusal-or-objection share."""
    if eval_id == "bfcl":
        return f"No call {shares['no_call']:.0f}% · format {shares['malformed_call'] + shares['call_then_cap']:.1f}%"
    return f"Format {shares['format']:.0f}% · refusal {shares['refusal_or_objection']:.0f}%"


def _minus(s: str) -> str:
    return s.replace("-", "−")


def composition(side: dict, model_id: str, eval_id: str, rows=ROWS) -> pd.DataFrame:
    """One row per drawn bar of one (model, eval) panel, in drawing order: the pool, its label, the class shares (% of its new
    failures, in stacking order), the headline shares, the change in pass rate with its interval, new failures and gains, and the runs."""
    out = []
    for i, name in enumerate(rows):
        if name is None:
            continue
        q = pool(side, eval_id, model_id, name)
        if q is None:
            continue
        rec = dict(slot=i, pool=name, label=FAMILY_LABEL.get(name) or R.LABEL.get(name, name), family=name in FAMILY_LABEL,
                   n_runs=len(q["run_ids"]), lost=q["lost"], gained=q["gained"], delta=q["delta_pp"][0], delta_lo=q["delta_pp"][1],
                   delta_hi=q["delta_pp"][2])
        for c, _col, _lab in CLASS_STYLE[eval_id]:
            rec[c] = q["shares"][c][0]
        rec["headline"] = headline(eval_id, {k: v[0] for k, v in q["shares"].items()})
        rec["detail"] = _minus(f"Δ {q['delta_pp'][0]:+.1f} pp · {q['lost']:,.0f} new failures")
        out.append(rec)
    return pd.DataFrame(out)


def composition_frame(side: dict) -> pd.DataFrame:
    """Every bar of both figures (Qwen3.5-4B's panels are the first rows of the models figure), as `composition` gives them."""
    frames = [composition(side, m, e).assign(model_id=m, eval_id=e) for m in MODELS for e in EVALS]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True)[["model_id", "eval_id"] + [c for c in frames[0].columns if c not in ("model_id", "eval_id")]]


# ---- the figures ------------------------------------------------------------------------------------------------------------------
def _slots(rows) -> tuple:
    """(y centre of each row, the panel's total height in bar rows): a bar row is 1, a gap row GAP_SLOT."""
    ys, y = [], 0.0
    for name in rows:
        h = GAP_SLOT if name is None else 1.0
        ys.append(y + h / 2)
        y += h
    return ys, y


def _text_width(fig, text: str, size: float, weight: str = "normal") -> float:
    """The rendered width of `text` in inches at `size` points."""
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


# THE TEXT SIZES (points). The row labels are the paper's tick-label size (`R.style()`: 18); the headline, detail and share text step down
# so two lines sit beside a bar; the models figure, four blocks tall, sets them one point smaller at a tighter row pitch.
SIZES = {"qwen": dict(label=18, headline=16, detail=14, share=14, tick=15, legend=15, title=20),
         "models": dict(label=18, headline=15, detail=13, share=13, tick=15, legend=15, title=20)}
TITLE_MIN = 16                     # a panel title too wide for its column steps down to no less than this


def _draw_panel(ax, comp: pd.DataFrame, eval_id: str, rows, labels: bool, sizes: dict, note: str | None = None):
    """One composition panel: per row a stacked horizontal bar of the class shares (white gaps between segments), the share in any segment
    of at least LABEL_MIN %, the headline and the change / count to the bar's right; row labels on the left when `labels`. A panel with
    no run (`note`) keeps its row labels and says so in place of the bars."""
    ys, total = _slots(rows)
    ax.set_xlim(0, 100)
    ax.set_ylim(total, 0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(R.MUTED)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=sizes["tick"], color=R.INK2)
    ax.set_xlabel("% of new failures", fontsize=sizes["tick"], color=R.INK2, labelpad=4)
    ax.tick_params(axis="x", length=4, color=R.MUTED)
    ax.tick_params(axis="y", length=0)
    named = [(y, name) for y, name in zip(ys, rows) if name is not None]
    ax.set_yticks([y for y, _ in named])
    ax.set_yticklabels([(FAMILY_LABEL.get(n) or R.LABEL.get(n, n)) if labels else "" for _, n in named], fontsize=sizes["label"], color=R.INK)
    if labels:
        for tl, (_, n) in zip(ax.get_yticklabels(), named):
            if n in FAMILY_LABEL:
                tl.set_fontweight("bold")
    if note:
        ax.text(0.0, total / 2, note, ha="left", va="center", fontsize=sizes["headline"], color=R.MUTED, gid=f"caperrors-none|{eval_id}")
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["bottom"].set_visible(False)
        return
    for xv in (25, 50, 75, 100):
        ax.axvline(xv, color=R.RULE, lw=1.0, zorder=0)
    by = {r.pool: r for r in comp.itertuples()}
    for y, name in named:
        r = by.get(name)
        if r is None:
            ax.text(1.5, y, "no run", ha="left", va="center", fontsize=sizes["detail"], color=R.MUTED)
            continue
        x = 0.0
        for c, col, _lab in CLASS_STYLE[eval_id]:
            v = float(getattr(r, c))
            if v <= 0:
                continue
            ax.barh(y, v, left=x, height=BAR_HEIGHT, color=col, edgecolor="white", linewidth=1.2, zorder=2,
                    gid=f"caperrors|{eval_id}|{name}|{c}|{v:.6f}")
            if v >= LABEL_MIN:
                ax.text(x + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=sizes["share"], color=ON_FILL.get(col, "white"), zorder=3)
            x += v
        ax.annotate(r.headline, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(10, 0.55 * sizes["headline"]),
                    textcoords="offset points", ha="left", va="center", fontsize=sizes["headline"], color=R.INK, annotation_clip=False,
                    gid=f"caperrors-headline|{eval_id}|{name}")
        ax.annotate(r.detail, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(10, -0.75 * sizes["detail"]),
                    textcoords="offset points", ha="left", va="center", fontsize=sizes["detail"], color=R.INK2, annotation_clip=False,
                    gid=f"caperrors-detail|{eval_id}|{name}")


def _legend(fig, eval_id: str, x: float, y: float, size: float, wrap: int):
    """The class legend of one evaluation, its top-left corner at figure fraction (x, y): the colours in stacking order, the text in the
    secondary ink, the evaluation named in its bold title."""
    handles = [Patch(facecolor=col, edgecolor="none", label=textwrap.fill(lab, wrap)) for _c, col, lab in CLASS_STYLE[eval_id]]
    lg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=1, frameon=False, fontsize=size,
                    title=f"{EVAL_LABEL[eval_id]}: class of each new failure", title_fontproperties={"weight": "semibold", "size": size + 1},
                    alignment="left", handlelength=1.2, handleheight=1.0, borderaxespad=0.0, borderpad=0.0, labelspacing=0.45)
    for t in lg.get_texts():
        t.set_color(R.INK2)
    lg.get_title().set_color(R.INK)
    return lg


def _legend_height(width: float, sizes: dict, wrap: int) -> float:
    """The taller of the two class legends, in inches, measured on a scratch canvas of the figure's width."""
    with AC.style_context():
        f = plt.figure(figsize=(width, 4))
        lgs = [_legend(f, e, 0.0, 1.0, sizes["legend"], wrap) for e in EVALS]
        f.canvas.draw()
        h = max(lg.get_window_extent().height for lg in lgs) / f.dpi
        plt.close(f)
    return h


def _layout(fig, panel_rows: list, sizes: dict, side_pad: float = 0.35) -> dict:
    """Axes positions in inches for a grid of panels: one column per eval (BFCL, BigCodeBench), one row per model block. The left
    column carries the row labels, each column the headline text to its bars' right; the bar width takes what the labels and text leave
    of the canvas width."""
    W = fig.get_figwidth()
    lab_w = max(_text_width(fig, FAMILY_LABEL.get(n) or R.LABEL.get(n, n), sizes["label"], "bold" if n in FAMILY_LABEL else "normal")
                for rows in panel_rows for n in rows if n) + 0.25
    txt_w = max(_text_width(fig, s, sizes["headline"]) for s in ("No call 100% · format 100.0%", "Format 100% · refusal 100%"))
    txt_w = max(txt_w, _text_width(fig, _minus("Δ −33.4 pp · 10,734 new failures"), sizes["detail"])) + 0.25
    bar_w = (W - lab_w - 2 * txt_w - side_pad) / 2
    if bar_w < 2.0:
        raise ValueError(f"a {W:.1f} in canvas leaves {bar_w:.1f} in per bar beside {lab_w:.1f} in of labels and 2 x {txt_w:.1f} in of text")
    return dict(lab_w=lab_w, txt_w=txt_w, bar_w=bar_w, x=[lab_w, lab_w + bar_w + txt_w + side_pad], col_w=[bar_w + txt_w + side_pad, bar_w + txt_w])


def qwen_figure(side: dict, name: str | None = "figA_qwen_capability_errors", width: float = AC.MODEL_WIDTH, pitch: float = 0.62,
                out_dirs=None, extra_dirs=()):
    """FIGURE `figA_qwen_capability_errors`: Qwen3.5-4B, BFCL (left) beside BigCodeBench (right), the three seeds of each fitted object
    pooled (the prompted conditions and the Refusal steer one run each). One bar per object and per family pool (bold), the classes as
    shares of its new failures; beside it the headline shares, the change in pass rate and the number of new failures. The class
    legends are on top, BFCL's over the BFCL panel. Saved (png, pdf, svg) as `name` when one is given; returned."""
    return _figure(side, ["qwen35_4b_it"], name, width, pitch, out_dirs, extra_dirs, SIZES["qwen"],
                   titles={("qwen35_4b_it", e): f"{EVAL_LABEL[e]} ({side['evals'][e]['n_items']:,} items)" for e in EVALS})


def models_figure(side: dict, name: str | None = "figA_models_capability_errors", width: float = AC.MODEL_WIDTH, pitch: float = 0.5,
                  out_dirs=None, extra_dirs=()):
    """FIGURE `figA_models_capability_errors`: the same composition for every model, one row of panels per model (Qwen3.5-4B pooled over
    its three seeds; one run per object elsewhere). Llama-3.1-8B has no BFCL run: its BFCL panel says so and carries the row labels."""
    titles = {(m, e): f"{R.MODEL_LABEL[m]}{' (seeds pooled)' if m == 'qwen35_4b_it' else ''}: {EVAL_LABEL[e]}" for m in MODELS for e in EVALS}
    return _figure(side, list(MODELS), name, width, pitch, out_dirs, extra_dirs, SIZES["models"], titles=titles)


def _rows_for(side: dict, model_id: str) -> list:
    """The panel rows a model draws: ROWS without the objects it has no run of on either eval (the Refusal steer outside Qwen3.5-4B)."""
    have = {q["pool"] for q in side["pools"] if q["model_id"] == model_id}
    rows = [r for r in ROWS if r is None or r in have]
    while rows and rows[-1] is None:
        rows.pop()
    return rows


def _figure(side, models, name, width, pitch, out_dirs, extra_dirs, sizes, titles):
    blocks = [(m, _rows_for(side, m)) for m in models]
    wrap = 58
    with AC.style_context():
        fig = plt.figure(figsize=(width, 10))
        L = _layout(fig, [rows for _m, rows in blocks], sizes)
        tsize = {}
        for (m, e), t in titles.items():                       # a title wider than its column steps down, never below TITLE_MIN
            sz = sizes["title"]
            while sz > TITLE_MIN and _text_width(fig, t, sz, "semibold") > L["col_w"][EVALS.index(e)]:
                sz -= 1
            tsize[(m, e)] = sz
        legend_h = _legend_height(width, sizes, wrap) + 0.3
        title_h, axis_h, block_gap = 0.6, 0.8, 0.35
        heights = [_slots(rows)[1] * pitch for _m, rows in blocks]
        H = legend_h + sum(heights) + len(blocks) * (title_h + axis_h) + (len(blocks) - 1) * block_gap
        fig.set_size_inches(width, H)
        top = H - legend_h
        axes = {}
        for (m, rows), h in zip(blocks, heights):
            y0 = top - title_h - h
            for ci, e in enumerate(EVALS):
                ax = fig.add_axes([L["x"][ci] / width, y0 / H, L["bar_w"] / width, h / H])
                comp = composition(side, m, e, rows)
                note = None if len(comp) else f"No {EVAL_LABEL[e]} run for {R.MODEL_LABEL[m]}"
                _draw_panel(ax, comp, e, rows, labels=(ci == 0), sizes=sizes, note=note)
                ax.set_title(titles[(m, e)], loc="left", fontsize=tsize[(m, e)], fontweight="semibold", color=R.INK, pad=10)
                ax.set_gid(f"caperrors-panel|{m}|{e}")
                axes[(m, e)] = ax
            top = y0 - axis_h - block_gap
        for ci, e in enumerate(EVALS):
            _legend(fig, e, L["x"][ci] / width, 1.0, sizes["legend"], wrap)
    if name:
        AC.save_figure(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig


def panel_axes(fig) -> dict:
    """{(model id, eval id): axes} of a composition figure, read from the axes' gids."""
    return {tuple(ax.get_gid().split("|")[1:]): ax for ax in fig.axes if str(ax.get_gid() or "").startswith("caperrors-panel|")}


def drawn_segments(fig) -> pd.DataFrame:
    """Every drawn bar segment of a composition figure: model, eval, pool, class, the share (%) its gid carries, and its drawn left edge
    and width in axis units, so a test or a reader can hold the canvas to the sidecar."""
    out = []
    for (m, e), ax in panel_axes(fig).items():
        for patch in ax.patches:
            gid = patch.get_gid() or ""
            if gid.startswith("caperrors|"):
                _t, ev, pool_, cls, v = gid.split("|")
                out.append(dict(model_id=m, eval_id=ev, pool=pool_, cls=cls, share=float(v), left=patch.get_x(), width=patch.get_width()))
    return pd.DataFrame(out)


# ---- the tables -------------------------------------------------------------------------------------------------------------------
def own_levels(side: dict, eval_id: str, model_id: str) -> dict:
    """{fitted object: its own unsteered level (%)}: the mean of the seed-matched unsteered runs its pooled runs were differenced against
    (for Qwen3.5-4B the three seeds' anchors; elsewhere the one anchor)."""
    by_id = {r["run_id"]: r for r in side["runs"]}
    out = {}
    for o in FITTED:
        q = pool(side, eval_id, model_id, o)
        if q is not None:
            out[o] = float(np.mean([100.0 * by_id[r]["anchor_k"] / by_id[r]["n"] for r in q["run_ids"]]))
    return out


def anchor_readings(side: dict, eval_id: str, model_id: str) -> list:
    """The unsteered readings behind a block's printed cells, for `registry_nb.seed_average`: one (seed, anchor run, level %, fitted)
    per steered run of the six fitted objects."""
    by_id = {r["run_id"]: r for r in side["runs"]}
    out = []
    for o in FITTED:
        q = pool(side, eval_id, model_id, o)
        for rid in (q["run_ids"] if q is not None else []):
            r = by_id[rid]
            out.append((r["seed"] or "", r["anchor_run_id"], 100.0 * r["anchor_k"] / r["n"], True))
    return out


def unsteered_level(side: dict, eval_id: str, model_id: str) -> float | None:
    """A block's first row, the unsteered pass rate (%), by the notebook's rule for every first row (`registry_nb.seed_average`): on
    Qwen3.5-4B the mean over the fit seeds of the unsteered run the most of that seed's arms were differenced against, on a model with
    one fit the run the most of its arms were differenced against. Every cell stays its arm's change against the arm's own unsteered
    run, so first row + cell is the arm's pass rate exactly where that run equals the first row and approximately elsewhere (on a
    model with one fit, the difference ReLU steer's paired serving view has its own unsteered run)."""
    return R.seed_average(anchor_readings(side, eval_id, model_id))


def _signed(v: float) -> str:
    s = f"{v:.1f}"
    return s if s.startswith("-") else "+" + s


def _delta_cell(v) -> str:
    return f"${_signed(v[0])}$ {{\\scriptsize $[{v[1]:.1f}, {v[2]:.1f}]$}}"


def _table(side: dict, eval_id: str, caption: str, label: str) -> str:
    if eval_id == "bfcl":
        heads = [_stack("No call", "(decl.)"), _stack("Malformed", "call"), _stack("Call,", "then cap"), _stack("Wrong", "call"),
                 _stack("Irrelevance", "call")]
    else:
        heads = ["Format", _stack("Wrong", "code"), _stack("Refusal or", "objection"), "Undecidable"]
    ncol = 3 + len(heads)
    lines = ["\\begin{table}[ht]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{center}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{l{'r' * (ncol - 1)}}}", "\\toprule",
             f" & & & \\multicolumn{{{len(heads)}}}{{c}}{{\\% of new failures}} \\\\", f"\\cmidrule(lr){{4-{ncol}}}",
             "\\multicolumn{1}{c}{Arm} & " + " & ".join([_PASS_HEAD, _NEW_HEAD] + heads) + " \\\\", "\\midrule"]
    first = True
    for m in MODELS:
        ref = unsteered_level(side, eval_id, m)
        if ref is None:
            continue
        if not first:
            lines.append("\\midrule")
        first = False
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{TABLE_BLOCK[m]}}}}} \\\\")
        lines.append(f"\\quad Unsteered & ${ref:.1f}$" + " &" * (ncol - 2) + " \\\\")
        for o in FITTED:
            q = pool(side, eval_id, m, o)
            if q is None:
                continue
            s = {k: v[0] for k, v in q["shares"].items()}
            if eval_id == "bfcl":
                cells = [f"{s['no_call']:.1f} ({s['no_call_declination']:.1f})", f"{s['malformed_call']:.1f}", f"{s['call_then_cap']:.1f}",
                         f"{s['wrong_call']:.1f}", f"{s['irrelevance_call']:.1f}"]
            else:
                cells = [f"{s['format']:.1f}", f"{s['wrong_code']:.1f}", f"{s['refusal_or_objection']:.1f}", f"{s['undecidable']:.1f}"]
            lines.append(f"\\quad {TABLE_ROW[o]} & {_delta_cell(q['delta_pp'])} & {q['lost']:,.0f} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}", "\\end{table}"]
    return "\n".join(lines)


def table_caperrors(side: dict, caption: str = CAPTION_BFCL, label: str = "tab:caperrors") -> str:
    """TABLE `tab:caperrors` (LaTeX): per model with a BFCL run, the unsteered pass rate, then per fitted object its pooled change in pass
    rate with the item-bootstrap interval, its new failures, and the five classes as % of them (no call with the declination share in
    parentheses, malformed call, call then cap, wrong call, irrelevance call)."""
    return _table(side, "bfcl", caption, label)


def table_caperrors_bcb(side: dict, caption: str = CAPTION_BCB, label: str = "tab:caperrors-bcb") -> str:
    """TABLE `tab:caperrors-bcb` (LaTeX): the same for BigCodeBench, four models, classes format (no code, malformed and capped with no
    program together), wrong code, refusal or objection, undecidable."""
    return _table(side, "bigcodebench", caption, label)


def seed_frame(side: dict, eval_id: str = "bfcl") -> pd.DataFrame:
    """Per run (Qwen3.5-4B's seeds apart): the change in pass rate with its interval, the new failures, and each class as % of them; the
    seed-to-seed spread the pooled intervals leave out."""
    rows = []
    for r in side["runs"]:
        if r["eval_id"] != eval_id:
            continue
        rec = dict(model_id=r["model_id"], object_label=r["object_label"], seed=r["seed"] or "-", view=r["view"], delta=r["delta_pp"][0],
                   delta_lo=r["delta_pp"][1], delta_hi=r["delta_pp"][2], new_failures=r["lost"], gains=r["gained"])
        for c, _col, _lab in CLASS_STYLE[eval_id]:
            rec[c] = 100.0 * r["lost_by_class"][c] / r["lost"] if r["lost"] else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)
