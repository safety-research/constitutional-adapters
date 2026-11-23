"""appendix_geometry_evenodd: the sign decomposition of the control objects (figA_control_evenodd), read from its committed sidecar.

    import appendix_geometry_evenodd as GE
    ge = GE.load_geometry_evenodd()                     # data/appendix_geometry_evenodd.json, refused by name when it breaks its contract
    GE.bind_problems(ge, objects, runs, judgments)      # every object, run and judgment the file pins, against the registry
    GE.register_payloads(ge)                            # every used run's per-unit payload, for the item-clustered intervals
    GE.figure_geometry_evenodd(ge)                      # the paired comparison: + sign a circle, - sign a diamond, joined per seed
    print(GE.table_geometry_evenodd(ge))                # the sign-independent and sign-dependent parts, per category, as LaTeX

THE QUESTION: the paper reads the control object theta_K as register, so subtracting it should remove register. If
theta_K were a refusal direction, its effect would flip with its sign. Each control object was therefore also served negated (alpha = -1:
the ReLU steer's write directions or the LoRA's B matrices times -1, every other tensor unchanged), and its effect split into a
sign-independent part (D(+) + D(-))/2 and a sign-dependent part (D(+) - D(-))/2. The Refusal steer (Arditi et al., 2024) at c = +0.5
and c = -0.5 is the positive control: a refusal direction, whose effect is almost all sign-dependent.

THE FIGURE is `appendix_common.paired_comparison_figure` on the sidecar's rows: objects are the control ReLU steer and the control LoRA
in their paper colours, and the Refusal steer in its own; the + sign is a circle, the - sign a diamond, joined per seed by a thin line in
the object's colour, so a seed's sign-independent part is the line's midpoint and its sign-dependent part half the line's rise; one
group per evaluation, then the category mean (each sign averaged over the category's evaluations within seed, then over seeds, by the
notebook's `category_means` and `pool_seeds`). y is D(X), the change from the unsteered model in percentage points, + better (defense =
100 - ASR; benign compliance = 100 - over-refusal label mean). Panels: static attacks, multi-turn attacks, benign compliance.

THE INTERVALS are the notebook's, built from the sidecar's per-unit payloads (`payloads`: each used run's verdict per goal or item, by
position in its evaluation's item set, which `item_sets` names by size and the sha256 of its sorted ids, and its units with a degenerate
reply), which `register_payloads` installs as a unit source of
`registry_nb` (UNIT_SOURCES, one per mode: all replies, coherent units); `registry_nb.row_units` holds each run's units to its reading,
so a payload that does not give its run's rate and n stops the render (UnitsMismatch). A mark's interval is its delta +/- 1.96
hypot(the run's own interval: Wilson on its goals for an attack success rate, the normal interval for an over-refusal label mean;
the anchor's runs item-clustered, `registry_nb.combo_se`), as a single-run delta of the main figures; a mean mark is the notebook's seed
pool (`registry_nb.pool_seeds`: the seeds' runs item-clustered across the seeds, the between-seed variance, the shared anchor once).
The sidecar's own `se`, `se_steered` and `anchor_se` are the lane's arithmetic (the Wilson interval for every rate, the anchor's runs
as independent draws), which the builder checked against the lane's tables and `sidecar_problems` holds; nothing draws them.

THE TABLE splits each category mean into its two parts with the notebook's seed pool (`parts`), the shared anchor counted correctly:
both signs of a seed are differenced against one unsteered anchor per evaluation, which cancels from the sign-dependent part and counts
once in the sign-independent part. D(+) and D(-) are each the seed pool of the figure's mean mark. The sign-independent part
(D(+) + D(-)) / 2 is the seed pool of that reading: both signs' runs item-clustered across the seeds and the signs, the between-seed
variance of the seeds' parts, the anchor once. The sign-dependent part (D(+) - D(-)) / 2 holds no anchor; its two signs' runs are
independent terms (unpaired, as the notebook keeps a delta's steered part and its anchor, and a transfer difference's two lanes), each
item-clustered across the seeds, plus the between-seed variance of the seeds' parts. One series (the Refusal steer) is the one-fit
rule. 95% = 1.96 se. Both signs use the same seeds: a seed with one sign only (the control ReLU steer's seed 1 on the benign
evaluations, whose alpha = -1 object has no run there) is left out of both, and named.

THE JUDGE is MiniMax-M3 throughout (StrongREJECT at 0.5 per goal; over-refusal C/P/I labels per item): the sidecar pins each run's
MiniMax-M3 judgment, and its builder (`data/build_appendix_geometry_evenodd.py`) recounted every run from the registry's items.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd

import registry_nb as R
import appendix_common as AC

SIDECAR = os.path.join(R.HERE, "data", "appendix_geometry_evenodd.json")
SCHEMA = "appendix_geometry_evenodd/2"
JUDGE = "minimax-m3"
FIGURE = "figA_control_evenodd"
#: the lane-fixed registration (`registry_nb.register_lane_fixed(LANE_FIGURES, lane_fixed_text(data))`): this figure and its table are
#: MiniMax-M3's under every render policy, the paper's included
LANE_FIGURES = "sign-decomposition figure and table"
Z95 = 1.959964
OBJECTS = ("control_relu", "control_lora", "arditi_allpos")     # the two control objects, then the positive control
FAMILY = {"relu": "control_relu", "lora": "control_lora", "refusal_steer": "arditi_allpos"}
SEEDED = ("control_relu", "control_lora")
CONDITION = {1: "plus", -1: "minus"}
CONDITIONS = ("plus", "minus")
DOSE = {"control_relu": {1: 1.0, -1: -1.0}, "control_lora": {1: 1.0, -1: -1.0}, "arditi_allpos": {1: 0.5, -1: -0.5}}
CONDITION_LABELS = {"plus": "+θ: α = +1 (Refusal steer c = +0.5)", "minus": "−θ: α = −1 (Refusal steer c = −0.5)"}
LABELS = {"arditi_allpos": "Refusal steer (positive control)"}
# The rows are already oriented (+ is better), so every category takes sign +1; the evals and y labels are the notebook's.
CATEGORIES = {c: (list(R.CATEGORIES[c][0]), 1, R.CATEGORIES[c][2]) for c in ("static", "multi-turn", "over-refusal")}
TITLES = {"static": "static", "multi-turn": "multi-turn", "over-refusal": "benign compliance"}
CATEGORY_LABEL = {"static": "Static defense", "multi-turn": "Multi-turn defense", "over-refusal": "Benign compliance"}
LAYOUT = [["static"], ["multi-turn", "over-refusal"]]
ROW_KEYS = ("family", "object", "seed", "category", "eval", "sign", "dose", "object_id", "run_id", "judgment_id", "anchor", "rate", "n_units",
            "n_ungraded", "delta", "se", "se_steered", "anchor_se", "ci_lo", "ci_hi")
COHERENT_KEYS = ("rate_coherent", "n_coherent", "delta_coherent", "se_coherent", "se_steered_coherent", "anchor_se_coherent", "ci_lo_coherent",
                 "ci_hi_coherent")
#: the unit sources a render registers (`register_payloads`), one per mode: a run's all-replies reading and its coherent recount are two
#: readings of one run, so their units are keyed apart (`registry_nb.unit_key` ends with the row's `judge_source`)
UNIT_SOURCE = {"all": "the sign-decomposition file's payload (all replies)", "coherent": "the sign-decomposition file's payload (coherent units)"}
#: each category's reading: an attack success rate (Wilson form) or an over-refusal label mean (direct form), as registry_nb reads them
METRIC = {"static": "asr", "multi-turn": "asr", "over-refusal": "over_refusal_label_mean"}
#: a payload unit's verdicts: a goal attacked (1) or not (0); an item's C/P/I label score
VERDICTS = {"static": (0.0, 1.0), "multi-turn": (0.0, 1.0), "over-refusal": (0.0, 0.5, 1.0)}
Z = 1.96                                                         # the notebook's interval multiplier (`registry_nb.pool_seeds`, `cell_interval`)


# ---- the sidecar --------------------------------------------------------------------------------------------------------------
def load_geometry_evenodd(path: str = SIDECAR) -> dict:
    """The sidecar, refused by name (ValueError listing every problem) when it breaks the contract `sidecar_problems` checks."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bad = sidecar_problems(data)
    if bad:
        raise ValueError(f"{os.path.basename(path)} refused:\n  " + "\n  ".join(bad))
    return data


def _close(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def sidecar_problems(data: dict) -> list:
    """Every way the sidecar breaks its contract, one phrase each (empty when it holds): the schema; the judge stamp (MiniMax-M3, the
    StrongREJECT threshold 0.5) on the file and a pinned judgment on every reading and anchor run; the provenance (every source file with
    a sha256, the run selection); each row's keys, category, sign and dose; the arithmetic (delta = -(rate - anchor), se = hypot(se_steered,
    anchor_se), ci = delta +/- 1.959964 se, all replies and coherent alike); one mark per (object, seed, eval, sign), both signs of a
    seed on one anchor; one object per (object, seed, sign), the -1 object the +1 object's negation; the unpaired cells named, not drawn;
    and the per-unit payloads (`payload_problems`)."""
    out = []
    if data.get("schema") != SCHEMA:
        return [f"schema {data.get('schema')!r}, expected {SCHEMA!r}"]
    j = data.get("judge") or {}
    if j.get("model") != JUDGE or (j.get("strongreject") or {}).get("judge_model") != JUDGE or (j.get("over_refusal") or {}).get("judge_model") != JUDGE:
        out.append(f"judge stamp {j.get('model')!r}: this sidecar is MiniMax-M3's (StrongREJECT and over-refusal alike)")
    if (j.get("strongreject") or {}).get("threshold") != 0.5:
        out.append(f"StrongREJECT threshold {(j.get('strongreject') or {}).get('threshold')!r}, expected 0.5")
    p = data.get("provenance") or {}
    files = p.get("source_files") or []
    if not files or any(not f.get("name") or len(str(f.get("sha256", ""))) != 64 for f in files):
        out.append("provenance must name every source file with its sha256")
    if (p.get("run_selection") or {}).get("registry_head") is None or not p.get("script") or not p.get("rules"):
        out.append("provenance lacks the run selection's registry head, the script or the rules")
    rows, anchors = data.get("rows") or [], data.get("anchors") or {}
    if not rows:
        return out + ["no rows"]
    cat_of = {e: c for c, (evals, _s, _y) in CATEGORIES.items() for e in evals}
    seen, pair_anchor = set(), {}
    for i, r in enumerate(rows):
        tag = f"row {i} ({r.get('object')} {r.get('seed') or '-'} {r.get('eval')} sign {r.get('sign')})"
        miss = [k for k in ROW_KEYS + COHERENT_KEYS if k not in r]
        if miss:
            out.append(f"{tag}: lacks {miss}")
            continue
        if FAMILY.get(r["family"]) != r["object"]:
            out.append(f"{tag}: family {r['family']!r} does not name object {r['object']!r}")
        if cat_of.get(r["eval"]) != r["category"]:
            out.append(f"{tag}: category {r['category']!r} is not the notebook's for {r['eval']!r}")
        if r["sign"] not in (1, -1) or DOSE.get(r["object"], {}).get(r["sign"]) != r["dose"]:
            out.append(f"{tag}: sign {r['sign']!r} and dose {r['dose']!r} do not agree")
        if (r["object"] in SEEDED) != bool(r["seed"]):
            out.append(f"{tag}: a control object is seeded, the Refusal steer is not")
        if not r["judgment_id"] or not r["run_id"] or len(str(r["object_id"])) != 64:
            out.append(f"{tag}: a reading pins its run, its 64-hex object id and its MiniMax-M3 judgment")
        a = anchors.get(r["anchor"])
        if a is None or a.get("eval") != r["eval"]:
            out.append(f"{tag}: anchor {r['anchor']!r} is not an anchor of {r['eval']}")
            continue
        if not (_close(r["delta"], -(r["rate"] - a["value"])) and _close(r["anchor_se"], a["se"])
                and _close(r["se"], math.hypot(r["se_steered"], r["anchor_se"]))
                and _close(r["ci_lo"], r["delta"] - Z95 * r["se"]) and _close(r["ci_hi"], r["delta"] + Z95 * r["se"])):
            out.append(f"{tag}: delta, se or interval does not follow from its rate, anchor and split se")
        if r["delta_coherent"] is not None and not (
                _close(r["anchor_se_coherent"], a["se_coherent"]) and _close(r["se_coherent"], math.hypot(r["se_steered_coherent"], r["anchor_se_coherent"]))
                and _close(r["ci_lo_coherent"], r["delta_coherent"] - Z95 * r["se_coherent"])
                and _close(r["ci_hi_coherent"], r["delta_coherent"] + Z95 * r["se_coherent"])):
            out.append(f"{tag}: the coherent delta's se or interval does not follow from its split se")
        key = (r["object"], r["seed"], r["eval"], r["sign"])
        if key in seen:
            out.append(f"{tag}: repeats a mark")
        seen.add(key)
        pair_anchor.setdefault(key[:3], set()).add(r["anchor"])
    for (o, s, e), keys in pair_anchor.items():
        if not all((o, s, e, sg) in seen for sg in (1, -1)):
            out.append(f"{o} {s or '-'} {e}: one sign only; an unpaired cell belongs under `unpaired`, not in `rows`")
        if len(keys) != 1:
            out.append(f"{o} {s or '-'} {e}: the two signs sit on different anchors {sorted(keys)}")
    pinned = {r["run_id"] for r in rows}
    for k, a in anchors.items():
        if set(a.get("runs_used", [])) & set(a.get("runs_left_out", [])) or len(a.get("judgment_ids", [])) != len(a.get("runs_used", [])) \
                or not all(a.get("judgment_ids", [])):
            out.append(f"anchor {k}: every run used pins its MiniMax-M3 judgment, and no run is both used and left out")
        pinned |= set(a.get("runs_used", []))
    if j.get("judgments_pinned") != len(pinned):
        out.append(f"the judge stamp counts {j.get('judgments_pinned')} pinned judgments; the rows and anchors pin {len(pinned)} runs")
    # A RUN READ FROM THE FILE IS ONE THE SIDECAR PINS AND THAT WAS FILED BETWEEN THE RUN SELECTION'S SNAPSHOT AND THE BUILD: the binding
    # waiver (`runs_read_from_file`) can then cover only such a run, never an arbitrary id the file lists
    sel_head = (p.get("run_selection") or {}).get("registry_head")
    build_head = (p.get("verdicts") or {}).get("registry_head_at_build")
    pinned_any = pinned | {x for a in anchors.values() for x in a.get("runs", []) + a.get("runs_left_out", [])}
    listed = set()
    for x in (p.get("run_selection") or {}).get("runs_not_in_snapshot") or []:
        rid, seq = x.get("run_id"), x.get("ingest_seq")
        if rid not in pinned_any or rid in listed or not isinstance(seq, int) or isinstance(seq, bool) \
                or not isinstance(sel_head, int) or not isinstance(build_head, int) or not sel_head < seq <= build_head:
            out.append(f"runs_not_in_snapshot {rid!r} (ingest {seq!r}): a run the sidecar pins, listed once, filed after the run "
                       f"selection's snapshot (head {sel_head!r}) and by the build (head {build_head!r})")
        listed.add(rid)
    objs = {(o["object"], o["seed"], o["sign"]): o for o in data.get("objects") or []}
    for r in rows:
        o = objs.get((r["object"], r["seed"], r["sign"]))
        if o is None or o["object_id"] != r["object_id"]:
            out.append(f"{r['object']} {r['seed'] or '-'} sign {r['sign']}: its object id is not the one `objects` lists")
    for (o, s, sg), x in objs.items():
        if sg == -1 and (x.get("parent_object_id") != (objs.get((o, s, 1)) or {}).get("object_id") or x.get("base_scale") != -1.0):
            out.append(f"{o} {s or '-'} sign -1: its object is not the +1 object negated")
    for u in data.get("unpaired") or []:
        if not u.get("why") or (u.get("object"), u.get("seed"), u.get("eval")) in pair_anchor:
            out.append(f"unpaired {u.get('object')} {u.get('seed')} {u.get('eval')}: named with its reason and absent from `rows`")
    return out + payload_problems(data)


def _payload_reading(data: dict, run_id: str, mode: str) -> tuple:
    """(rate in pp, n, graded units, ungraded goals) of a run's payload: all replies count every unit of its evaluation's item set, an
    ungraded goal (null) as not attacked; the coherent recount the graded units without a degenerate reply."""
    p = data["payloads"][run_id]
    v = p["v"]
    graded = [k for k, x in enumerate(v) if x is not None]
    if mode == "coherent":
        deg = set(p["degenerate"])
        use = [k for k in graded if k not in deg]
        return (100.0 * sum(v[k] for k in use) / len(use) if use else float("nan")), len(use), len(graded), len(v) - len(graded)
    return 100.0 * sum(v[k] for k in graded) / len(v), len(v), len(graded), len(v) - len(graded)


def payload_problems(data: dict) -> list:
    """Every way the per-unit payloads break the contract, one phrase each: one item set per evaluation the rows read (its size and the
    sha256 of its sorted ids); a payload for exactly the runs a reading or an anchor uses, on its run's evaluation, one verdict per item (a goal 0 or 1,
    an item's label score 0, 0.5 or 1; null only for a StrongREJECT goal filed but not graded), its degenerate units graded positions;
    and the arithmetic: each reading's rate and n, all replies and coherent, and each anchor's value and coherent value are its runs'
    payload readings exactly."""
    out = []
    sets, pays = data.get("item_sets"), data.get("payloads")
    if not isinstance(sets, dict) or not isinstance(pays, dict):
        return ["no per-unit payloads (`item_sets` and `payloads`): the item-clustered intervals have nothing to read"]
    cat_of = {e: c for c, (evals, _s, _y) in CATEGORIES.items() for e in evals}
    rows, anchors = data.get("rows") or [], data.get("anchors") or {}
    for e in sorted({r.get("eval") for r in rows}):
        s = sets.get(e)
        n = s.get("n_items") if isinstance(s, dict) else None
        if not isinstance(n, int) or isinstance(n, bool) or n < 2 or not re.fullmatch(r"[0-9a-f]{64}", str(s.get("ids_sha256", ""))):
            out.append(f"item_sets[{e!r}]: one item set per evaluation the rows read, its size and the sha256 of its sorted ids")
    eval_of = {r.get("run_id"): r.get("eval") for r in rows}
    for a in anchors.values():
        eval_of.update({x: a.get("eval") for x in a.get("runs_used", [])})
    if set(pays) != set(eval_of):
        out.append(f"payloads: {len(set(eval_of) - set(pays))} used run(s) without one, {len(set(pays) - set(eval_of))} for a run no reading "
                   "or anchor uses")
    ok = set()
    for rid, p in pays.items():
        e = eval_of.get(rid)
        n = sets.get(e).get("n_items") if isinstance(sets.get(e), dict) else None
        if e is None or not isinstance(n, int) or p.get("eval") != e or not isinstance(p.get("v"), list) or len(p["v"]) != n:
            out.append(f"payload {rid}: one verdict per item of its run's evaluation's item set ({e})")
            continue
        allowed = VERDICTS[cat_of[e]]
        bad = [x for x in p["v"] if not (x is None and METRIC[cat_of[e]] == "asr") and x not in allowed]
        graded = {k for k, x in enumerate(p["v"]) if x is not None}
        deg = p.get("degenerate")
        if bad or not isinstance(deg, list) or deg != sorted(set(deg)) or not set(deg) <= graded:
            out.append(f"payload {rid}: verdicts in {allowed} (null only for an ungraded StrongREJECT goal) and degenerate units among its "
                       "graded positions")
            continue
        ok.add(rid)
    for i, r in enumerate(rows):
        rid = r.get("run_id")
        if rid not in ok:
            continue
        y, n, g, ung = _payload_reading(data, rid, "all")
        if not (_close(y, r.get("rate")) and g == r.get("n_units") and ung == r.get("n_ungraded")):
            out.append(f"row {i} ({rid}): its payload reads {y:.4f} over {g} graded units and {ung} ungraded, not its rate {r.get('rate')} over "
                       f"{r.get('n_units')} and {r.get('n_ungraded')}")
        if r.get("rate_coherent") is not None:
            yc, nc, _g, _u = _payload_reading(data, rid, "coherent")
            if not (_close(yc, r.get("rate_coherent")) and nc == r.get("n_coherent")):
                out.append(f"row {i} ({rid}): its payload's coherent reading {yc:.4f} over {nc} is not {r.get('rate_coherent')} over "
                           f"{r.get('n_coherent')}")
    for k, a in anchors.items():
        for mode, used, val in (("all", a.get("runs_used", []), a.get("value")), ("coherent", a.get("runs_used_coherent", []), a.get("value_coherent"))):
            if not used or not all(x in ok for x in used):
                continue
            ys = [_payload_reading(data, x, mode)[0] for x in used]
            if val is None or not _close(float(np.mean(ys)), val):
                out.append(f"anchor {k}: its {mode} value {val} is not the mean of its runs' payload readings ({float(np.mean(ys)):.4f})")
        if set(a.get("runs_used_coherent", [])) - set(a.get("runs_used", [])):
            out.append(f"anchor {k}: a run used for the coherent anchor that the all-replies anchor does not use")
    return out


# ---- the per-unit payloads: the notebook's unit source ----------------------------------------------------------------------------
_PAYLOADS: dict = {}       # (run id, mode) -> (the payload's signature, its reading's units: judge_rules.ReadingUnits)
_UNIT_ROW_CACHE: dict = {}  # (run id, mode, signature) -> the unit row registered in registry_nb._UNIT_ROWS
_ROW_OF: dict = {}          # (run id, mode) -> the unit row of the payload registered last


def payload_units(data: dict, run_id: str, mode: str = "all"):
    """The cluster units (`judge_rules.ReadingUnits`) of a run's reading from its payload: one unit per item of its evaluation, keyed by
    its position in the item set ("<eval>:<k>", one namespace for every run of the file on that evaluation), b = 1; all replies every
    item (an ungraded goal a = 0, as the reading counts it), the coherent recount the graded items without a degenerate reply. `present`
    is the whole item set (the alignment guard's ids)."""
    JR = R._judge_rules()
    p = data["payloads"][run_id]
    ids = [f"{p['eval']}:{k}" for k in range(data["item_sets"][p["eval"]]["n_items"])]
    deg = set(p["degenerate"]) if mode == "coherent" else set()
    keep = [k for k, x in enumerate(p["v"]) if (mode == "all" or x is not None) and k not in deg]
    a = pd.Series([float(p["v"][k] or 0.0) for k in keep], index=[ids[k] for k in keep], dtype=float)
    return JR.ReadingUnits(a, pd.Series(1.0, index=a.index), None, frozenset(ids))


def _unit_source(mode: str):
    def units(row):
        hit = _PAYLOADS.get((str(row["run_id"]), mode))
        if hit is None:
            raise RuntimeError(f"{row['run_id']}: read from the sign-decomposition file's payload, but no payload of it was registered "
                               "(register_payloads)")
        return hit[1]
    return units


for _mode, _src in UNIT_SOURCE.items():
    R.UNIT_SOURCES[_src] = _unit_source(_mode)


def _judgments(data: dict) -> dict:
    """{run id: the MiniMax-M3 judgment the file pins for it} over the readings and the anchors' runs."""
    out = {r["run_id"]: r["judgment_id"] for r in data["rows"]}
    for a in data["anchors"].values():
        out.update(dict(zip(a["runs_used"], a["judgment_ids"])))
    return out


def register_payloads(data: dict) -> int:
    """Install every used run's payload as `registry_nb`'s unit source (both modes) and register each run's unit row (`unit_row`) in
    `registry_nb._UNIT_ROWS`, where the seed pools and the anchor mixtures read it by its unit key; returns the number of runs. Another
    payload of the same run (another file loaded) replaces it, and the units cached from the earlier one are dropped with the pools
    built on them. Idempotent: every reader of the intervals calls it, so a pool cache reset in between costs nothing."""
    jm = _judgments(data)
    reading = {r["run_id"]: r for r in data["rows"]}
    obj = {r["run_id"]: r["object"] for r in data["rows"]}
    for rid, p in data["payloads"].items():
        sig = (p["eval"], tuple(p["v"]), tuple(p["degenerate"]), jm.get(rid))
        for mode, src in UNIT_SOURCE.items():
            prev = _PAYLOADS.get((rid, mode))
            if prev is None or prev[0] != sig:
                if prev is not None:
                    stale = [k for k in R._UNITS if k.startswith(rid + "|") and k.endswith("|" + src)]
                    for k in stale:
                        R._UNITS.pop(k, None)
                    R._POOLS.clear()
                    R._SEED_POOLS.clear()
                _PAYLOADS[(rid, mode)] = (sig, payload_units(data, rid, mode))
            row = _UNIT_ROW_CACHE.get((rid, mode, sig))
            if row is None:
                row = unit_row(data, rid, mode, reading.get(rid), obj.get(rid, "none"), jm.get(rid))
                _UNIT_ROW_CACHE[(rid, mode, sig)] = row
            R._UNIT_ROWS[R.unit_key(row)] = row
            _ROW_OF[(rid, mode)] = row
    return len(data["payloads"])


def unit_row(data: dict, run_id: str, mode: str, reading: dict | None = None, object_label: str = "none", judgment_id: str | None = None):
    """The row `registry_nb` pools a run's reading by (`unit_key`, `row_units`, `own_se`): its run, metric (the attack success rate or
    the over-refusal label mean), pinned judgment, threshold, this file's unit source, its reading (a reading's stored rate and n; an
    anchor run's payload reading) and its own interval in pp (Wilson on its goals for a rate; the normal interval for a label mean)."""
    JR = R._judge_rules()
    e = data["payloads"][run_id]["eval"]
    cat = {e_: c for c, (evals, _s, _y) in CATEGORIES.items() for e_ in evals}[e]
    u = payload_units(data, run_id, mode)
    y_u, n_u = float(u.a.sum() / u.b.sum()), float(u.b.sum())
    if reading is not None:
        y = (reading["rate_coherent"] if mode == "coherent" else reading["rate"])
        n = (reading["n_coherent"] if mode == "coherent" else reading["n_units"] + reading["n_ungraded"])
        y, n = (np.nan, np.nan) if y is None else (float(y) / 100.0, float(n))
    else:
        y, n = y_u, n_u
    if METRIC[cat] == "asr":
        lo, hi = JR.wilson(float(u.a.sum()), n_u)
    else:
        lo, hi = JR.normal_mean_interval(u.a.to_numpy(float))
    return pd.Series(dict(run_id=run_id, metric_id=METRIC[cat], judgment_id=judgment_id or "", threshold=(0.5 if METRIC[cat] == "asr" else -1.0),
                          bon_n=None, judge_source=UNIT_SOURCE[mode], value=y, n=n, eval_id=e, model_id=R.MODEL_ID, object_label=object_label,
                          item_set_id=f"appendix_geometry_evenodd:{e}", lo=100.0 * lo, hi=100.0 * hi))


def _key(run_id: str, mode: str) -> tuple:
    """(unit key, own se in pp) of a run's reading, from the payload registered last (`register_payloads` first)."""
    row = _ROW_OF[(run_id, mode)]
    return R.unit_key(row), R.own_se(row)


def anchor_weights(data: dict, anchor: str, mode: str = "all") -> dict:
    """{unit key: weight} of an anchor's runs (each 1 / K: the anchor is their mean), the runs used for the mode."""
    a = data["anchors"][anchor]
    runs = a["runs_used_coherent"] if mode == "coherent" else a["runs_used"]
    return {_key(x, mode)[0]: 1.0 / len(runs) for x in runs}


def rows(data: dict, coherent: bool = False) -> pd.DataFrame:
    """The tidy rows `appendix_common` draws: object, seed, category, eval, condition (plus / minus), delta (oriented, + better), its
    interval by the notebook's rule (ci_lo, ci_hi: delta +/- 1.96 hypot(the run's own se, the anchor runs' item-clustered se)), n (units
    behind the reading), and the delta's parts the mean marks' seed pool reads (`registry_nb.delta_parts`' form: `se_steer`, the run's
    own se; `steer_terms`, its unit key; `anchor_terms` and `split_anchor_terms`, the anchor's runs with their weights). `coherent` reads
    the union5-coherent recount instead of all replies. Registers the payloads first (`register_payloads`)."""
    register_payloads(data)
    mode = "coherent" if coherent else "all"
    sfx = "_coherent" if coherent else ""
    out = []
    for r in data["rows"]:
        d = r["delta" + sfx]
        k, se_s = _key(r["run_id"], mode)
        aw = anchor_weights(data, r["anchor"], mode)
        se = math.hypot(se_s, R.combo_se(aw)) if d is not None else np.nan
        out.append(dict(object=r["object"], seed=r["seed"], category=r["category"], eval=r["eval"], condition=CONDITION[r["sign"]],
                        delta=np.nan if d is None else float(d), ci_lo=np.nan if d is None else d - Z * se,
                        ci_hi=np.nan if d is None else d + Z * se, n=r["n_coherent"] if coherent else r["n_units"],
                        se_steer=se_s, anchor_terms=aw, steer_terms={k: 1.0}, split_anchor_terms=dict(aw)))
    return pd.DataFrame(out, columns=["object", "seed", "category", "eval", "condition", "delta", "ci_lo", "ci_hi", "n", "se_steer",
                                      "anchor_terms", "steer_terms", "split_anchor_terms"])


# ---- the sign decomposition ---------------------------------------------------------------------------------------------------
PARTS = ("Dp", "Dn", "even", "odd")


def readings(data: dict, coherent: bool = False) -> pd.DataFrame:
    """One row per (object, seed, eval) with both signs: Dp = D(+), Dn = D(-) (oriented, + better), the two steered runs' unit keys
    (`key_p`, `key_n`) and the shared anchor's runs with their weights (`anchor`, {unit key: weight}); the payloads registered. A cell
    whose coherent recount is missing for a sign is left out of the coherent readings."""
    register_payloads(data)
    mode = "coherent" if coherent else "all"
    sfx = "_coherent" if coherent else ""
    by = {(r["object"], r["seed"], r["eval"], r["sign"]): r for r in data["rows"]}
    out = []
    for (o, s, e, sg), p in by.items():
        if sg != 1:
            continue
        n = by[(o, s, e, -1)]
        if p["delta" + sfx] is None or n["delta" + sfx] is None:
            continue
        out.append(dict(object=o, seed=s, category=p["category"], eval=e, Dp=float(p["delta" + sfx]), Dn=float(n["delta" + sfx]),
                        key_p=_key(p["run_id"], mode)[0], key_n=_key(n["run_id"], mode)[0], anchor=anchor_weights(data, p["anchor"], mode)))
    return pd.DataFrame(out, columns=["object", "seed", "category", "eval", "Dp", "Dn", "key_p", "key_n", "anchor"])


def _value(g: pd.DataFrame, part: str) -> pd.Series:
    return {"Dp": g.Dp, "Dn": g.Dn, "even": (g.Dp + g.Dn) / 2, "odd": (g.Dp - g.Dn) / 2}[part]


def seed_pool(g: pd.DataFrame, part: str) -> tuple:
    """(mean, se, how) of `part` over the readings `g` of one object (a category's evaluations, or one evaluation): each seed's unweighted
    mean over its evaluations in `g`, then the seeds' equal-weight mean, its se by the notebook's seed pool (`registry_nb.seed_pool`;
    the module docstring's THE TABLE): D(+) and D(-) the mean mark's pool; the sign-independent part the pool of both signs' runs, item-
    clustered across the seeds and the signs, the shared anchor once; the sign-dependent part no anchor, its two signs' runs independent
    terms each item-clustered across the seeds, plus the between-seed variance of the seeds' parts. One seed is the one-fit rule."""
    M, PCI = R.SeedMember, R._pooled_ci()
    o = str(g.object.iloc[0])
    ys, full, half = [], {"p": [], "n": [], "even": []}, {"p": [], "n": []}
    anchors = []
    for s in dict.fromkeys(g.seed):
        gs = g[g.seed.eq(s)]
        k = len(gs)
        ys.append(float(_value(gs, part).mean()))
        full["p"].append(tuple((r.eval, ((r.key_p, 1.0 / k),)) for r in gs.itertuples()))
        full["n"].append(tuple((r.eval, ((r.key_n, 1.0 / k),)) for r in gs.itertuples()))
        full["even"].append(tuple((r.eval, tuple(sorted(((r.key_p, 0.5 / k), (r.key_n, 0.5 / k))))) for r in gs.itertuples()))
        half["p"].append(tuple((r.eval, ((r.key_p, 0.5 / k),)) for r in gs.itertuples()))
        half["n"].append(tuple((r.eval, ((r.key_n, 0.5 / k),)) for r in gs.itertuples()))
        anchors.append(tuple((r.eval, tuple(sorted((k_, w_ / k) for k_, w_ in r.anchor.items()))) for r in gs.itertuples()))
    seeds = [f"{o}|{s}" for s in dict.fromkeys(g.seed)]
    what = "sign-decomposition seed pool"
    if part in ("Dp", "Dn", "even"):
        steer = full[{"Dp": "p", "Dn": "n", "even": "even"}[part]]
        res = R.seed_pool([M(k_, y, st, an, np.nan) for k_, y, st, an in zip(seeds, ys, steer, anchors)], what=what)
        return float(np.mean(ys)), res.se, res.method
    sides = [R.seed_pool([M(k_, y, st, (), np.nan) for k_, y, st in zip(seeds, ys, half[sg])], what=what, anchor=False) for sg in ("p", "n")]
    v = sides[0].v_items + sides[1].v_items + PCI.fit_var(ys)
    how = sides[0].method if sides[0].method == sides[1].method else f"{sides[0].method}; {sides[1].method}"
    return float(np.mean(ys)), (float(np.sqrt(v)) if np.isfinite(v) else np.nan), f"{how} (two signs unpaired)"


def seed_parts(data: dict, coherent: bool = False) -> pd.DataFrame:
    """One row per (object, seed, category, eval): Dp = D(+), Dn = D(-), even = (Dp + Dn) / 2 and odd = (Dp - Dn) / 2, each with its se
    by the notebook's rule on one reading (`seed_pool` over one seed and one evaluation): Dp and Dn hypot(the run's own se, the anchor's
    item-clustered se); even the two runs item-clustered with weights 1/2 and the anchor once; odd sqrt(se_p^2 / 4 + se_n^2 / 4), the
    shared anchor cancelling."""
    rd = readings(data, coherent)
    out = []
    for _, r in rd.iterrows():
        rec = dict(object=r.object, seed=r.seed, category=r.category, eval=r["eval"])
        for q in PARTS:
            v, se, _how = seed_pool(rd.loc[[r.name]], q)
            rec.update({q: v, f"{q}_se": se})
        out.append(rec)
    return pd.DataFrame(out, columns=["object", "seed", "category", "eval"] + [c for q in PARTS for c in (q, f"{q}_se")])


def eval_parts(data: dict, coherent: bool = False) -> pd.DataFrame:
    """One row per (object, eval): Dp, Dn, even and odd pooled over seeds with 95% half-widths (`*_ci`, 1.96 se) by the notebook's seed
    pool (`seed_pool`), the per-seed values beside them (`*_seeds`), and the unsteered anchor rate averaged over the seeds (`anchor`, ASR
    or over-refusal %). The per-evaluation numbers the appendix text quotes (the Refusal steer has one run per sign, so its half-width is
    its sampling error, with its anchor's where the part holds it)."""
    rd = readings(data, coherent)
    sfx = "_coherent" if coherent else ""
    anc = {(r["object"], r["seed"], r["eval"]): data["anchors"][r["anchor"]]["value" + sfx] for r in data["rows"]}
    out = []
    for (o, e), g in rd.groupby(["object", "eval"], sort=False):
        rec = dict(object=o, category=g.category.iloc[0], eval=e, seeds=",".join(s for s in g.seed if s), n_seeds=len(g),
                   anchor=float(np.mean([anc[(o, s, e)] for s in g.seed])))
        for q in PARTS:
            v, se, _how = seed_pool(g, q)
            rec.update({q: v, f"{q}_ci": Z * se, f"{q}_seeds": " ".join(f"{s}:{x:+.1f}" for s, x in zip(g.seed, _value(g, q)) if s)})
        out.append(rec)
    order = {o: i for i, o in enumerate(OBJECTS)}
    eorder = {e: i for i, e in enumerate(e for c in CATEGORIES.values() for e in c[0])}
    df = pd.DataFrame(out)
    return df.sort_values(["object", "eval"], key=lambda s: s.map(order if s.name == "object" else eorder)).reset_index(drop=True)


def category_parts(data: dict, coherent: bool = False) -> pd.DataFrame:
    """One row per (object, category): Dp, Dn, even and odd as seed-pooled category means (within seed the unweighted mean over the
    category's evaluations, then the seeds' equal-weight mean) with 95% half-widths (`*_ci`, 1.96 se) by the notebook's seed pool
    (`seed_pool`; D(+) and D(-) are the figure's mean marks), how each was obtained (`*_how`), and the per-seed values. The Refusal steer
    is one series."""
    rd = readings(data, coherent)
    out = []
    for (o, c), g in rd.groupby(["object", "category"], sort=False):
        rec = dict(object=o, category=c, n_evals=int(g.groupby("seed")["eval"].nunique().min()))
        seeds = sorted(g.seed.unique())
        rec["seeds"] = ",".join(s for s in seeds if s)
        rec["n_seeds"] = len(seeds)
        for q in PARTS:
            v, se, how = seed_pool(g, q)
            per = _value(g, q).groupby(g.seed).mean()
            rec.update({q: v, f"{q}_ci": Z * se, f"{q}_how": how, f"{q}_seeds": " ".join(f"{s}:{x:+.1f}" for s, x in per.items() if s)})
        out.append(rec)
    order = {o: i for i, o in enumerate(OBJECTS)}
    corder = {c: i for i, c in enumerate(CATEGORIES)}
    return pd.DataFrame(out).sort_values(["object", "category"], key=lambda s: s.map(order if s.name == "object" else corder)).reset_index(drop=True)


def unsteered_levels(data: dict, coherent: bool = False) -> dict:
    """{category: the unsteered model's level of the category's quantity} (defense or benign compliance, %), for the table's first row,
    as a seed average: for each control object and seed, 100 minus each evaluation's anchor rate (the one both signs are differenced
    against), averaged over the category's evaluations; then averaged over every (object, seed) the category has."""
    sfx = "_coherent" if coherent else ""
    a = data["anchors"]
    r = pd.DataFrame([x for x in data["rows"] if x["sign"] == 1 and x["object"] in SEEDED])
    r["lev"] = [100.0 - a[k]["value" + sfx] for k in r.anchor]
    per = r.groupby(["object", "seed", "category"])["lev"].mean().reset_index()
    return {c: float(per.loc[per.category.eq(c), "lev"].mean()) for c in CATEGORIES}


def coherent_moves(data: dict) -> dict:
    """How far dropping units with a degenerate reply moves what the figure draws: the largest |coherent - all replies| over the
    per-seed marks and over the drawn category means, each with where it is."""
    a, c = rows(data), rows(data, coherent=True)
    k = ["object", "seed", "category", "eval", "condition"]
    m = a.merge(c, on=k, suffixes=("", "_c"))
    m["move"] = (m.delta_c - m.delta).abs()
    w = m.loc[m.move.idxmax()]
    pa, pc = AC.paired_means(a, CATEGORIES), AC.paired_means(c, CATEGORIES)
    mm = pa.merge(pc, on=["object", "condition", "category"], suffixes=("", "_c"))
    mm["move"] = (mm.y_c - mm.y).abs()
    wm = mm.loc[mm.move.idxmax()]
    sign = {"plus": "+θ", "minus": "−θ"}
    return dict(marks=float(w.move), marks_where=f"{R.LABEL.get(w.object, w.object)} {w.seed or ''} {R.EVAL_LABEL[w['eval']]} {sign[w.condition]}".replace("  ", " "),
                means=float(wm.move), means_where=f"{R.LABEL.get(wm.object, wm.object)} {TITLES[wm.category]} {sign[wm.condition]}")


# ---- the registry binding -----------------------------------------------------------------------------------------------------
def runs_sql(data: dict) -> str:
    """The SQL that reads every run the sidecar pins (readings and anchor runs, used or left out) from the registry's run table."""
    ids = sorted({r["run_id"] for r in data["rows"]} | {x for a in data["anchors"].values() for x in a["runs"]})
    return ("SELECT run_id, object_id, eval_id, model_id, lifecycle_state FROM run WHERE run_id IN ("
            + ",".join("'" + i.replace("'", "''") + "'" for i in ids) + ")")


def lane_fixed_text(data: dict) -> str:
    """The judge as the render stamp names it (`registry_nb.register_lane_fixed`): the model, StrongREJECT at the sidecar's threshold per
    goal and the over-refusal labels per item, then the head of each judge id the sidecar pins."""
    j = data["judge"]
    return (f"{j['model']}: StrongREJECT at {j['strongreject']['threshold']} per goal and over-refusal labels per item, judges "
            + ", ".join(i[:8] for i in j["judge_ids"]))


def judgments_sql(data: dict) -> str:
    """The SQL that reads every MiniMax-M3 judgment the sidecar pins, with its status and judge model."""
    ids = sorted({r["judgment_id"] for r in data["rows"]} | {x for a in data["anchors"].values() for x in a["judgment_ids"]})
    return ("SELECT j.judgment_id, j.run_id, j.status, g.judge_model FROM judgment j JOIN judge g ON g.judge_id = j.judge_id "
            "WHERE j.judgment_id IN (" + ",".join("'" + i.replace("'", "''") + "'" for i in ids) + ")")


def runs_read_from_file(data: dict, runs: pd.DataFrame, head: int | None = None) -> list:
    """The pinned runs this source does not hold that the sidecar names as filed after its run selection's snapshot
    (provenance.run_selection.runs_not_in_snapshot: run id and live ingest sequence), where the source is older than their filing (its
    head, default `runs.attrs['head']`, below their ingest sequence): a render from that snapshot reads them from the file and names
    them. A source that holds them, or is newer than them and still lacks them, binds them as every other run."""
    head = runs.attrs.get("head") if head is None else head
    have = set(runs.run_id)
    later = {x["run_id"]: int(x["ingest_seq"]) for x in (data.get("provenance") or {}).get("run_selection", {}).get("runs_not_in_snapshot", [])}
    return sorted(rid for rid, seq in later.items() if rid not in have and head is not None and int(head) < seq)


def bind_problems(data: dict, objects: pd.DataFrame, runs: pd.DataFrame, judgments: pd.DataFrame | None = None,
                  head: int | None = None) -> list:
    """Every pinned identity the registry does not hold as the sidecar claims it, one phrase each (empty when all bind). `objects` is
    `R.OBJECTS_SQL`'s frame: each object's paper label, the +1 object's seed, the -1 object's parent (the +1 object of its seed).
    `runs` is `runs_sql`'s: each reading's run is a complete qwen35_4b_it run of its object on its evaluation, each anchor run an
    unsteered run of its evaluation. `judgments` (`judgments_sql`'s; None to skip, as on a frame cut before the MiniMax-M3 re-judge,
    which holds none of them): each pinned judgment is a current, complete MiniMax-M3 judgment of its run. A run filed after the run
    selection's snapshot, on a source older than it (`runs_read_from_file`, `head` as there), is read from the file and is no problem."""
    out = []
    from_file = set(runs_read_from_file(data, runs, head))
    o = objects.set_index("object_id")
    for x in data["objects"]:
        tag = f"{x['object']} {x['seed'] or '-'} sign {x['sign']:+d} ({x['object_id'][:12]})"
        if x["object_id"] not in o.index:
            out.append(f"{tag}: not in the registry's object table")
            continue
        r = o.loc[x["object_id"]]
        seed = r["fit_replicate_label"]
        seed = "" if seed is None or (isinstance(seed, float) and math.isnan(seed)) else str(seed)
        parents = json.loads(r["parent_object_ids_json"]) if isinstance(r["parent_object_ids_json"], str) and r["parent_object_ids_json"] else []
        if str(r["paper_label"]) != x["object"]:
            out.append(f"{tag}: the registry labels it {r['paper_label']!r}")
        elif x["sign"] == 1 and seed != x["seed"]:
            out.append(f"{tag}: the registry's seed is {seed!r}")
        elif x["sign"] == -1 and parents != [x["parent_object_id"]]:
            out.append(f"{tag}: the registry's parent is {parents}, not {x['parent_object_id']}")
    rr = runs.set_index("run_id")
    for r in data["rows"]:
        if r["run_id"] in from_file:
            continue
        x = rr.loc[r["run_id"]] if r["run_id"] in rr.index else None
        if x is None or x.object_id != r["object_id"] or x.eval_id != r["eval"] or x.model_id != "qwen35_4b_it" \
                or x.lifecycle_state not in ("judged", "ingested"):
            out.append(f"run {r['run_id']}: not a complete qwen35_4b_it run of {r['object_id'][:12]} on {r['eval']} in this source")
    for k, a in data["anchors"].items():
        for rid in a["runs"]:
            if rid in from_file:
                continue
            x = rr.loc[rid] if rid in rr.index else None
            if x is None or x.object_id != "none" or x.eval_id != a["eval"]:
                out.append(f"anchor {k}: run {rid} is not an unsteered run of {a['eval']} in this source")
    if judgments is not None:
        jj = judgments.set_index("judgment_id")
        pins = [(r["judgment_id"], r["run_id"]) for r in data["rows"]] + \
               [(j, rid) for a in data["anchors"].values() for j, rid in zip(a["judgment_ids"], a["runs_used"])]
        for jid, rid in sorted(set(pins)):
            x = jj.loc[jid] if jid in jj.index else None
            if x is None or x.run_id != rid or x.judge_model != JUDGE or x.status != "complete":
                out.append(f"judgment {jid} of {rid}: not a complete MiniMax-M3 judgment of that run in the registry"
                           + (f" (status {x.status})" if x is not None else ""))
    return out


# ---- the table and the figure -------------------------------------------------------------------------------------------------
def _pm(v, ci) -> str:
    return "--" if v is None or not np.isfinite(v) else f"${v:+.1f}\\pm{ci:.1f}$"


TABLE_CAPTION = (r"Sign decomposition of the control objects, by category (MiniMax-M3 judge). First row: the unsteered value of each "
                 r"column's quantity (defense $=100-\mathrm{ASR}$; benign compliance $=100-$ over-refusal), for scale. Other cells: $D(X)$, "
                 r"the change from the unsteered model in pp (higher is better), and its sign-independent part $\tfrac12[D(+\theta)+D(-\theta)]$ "
                 r"and sign-dependent part $\tfrac12[D(+\theta)-D(-\theta)]$, both signs on one unsteered baseline per evaluation and seed. A "
                 r"category value averages its evaluations within seed, then the seeds. $\pm$: normal 95\,\% interval: the seeds' runs "
                 r"clustered by goal or item across the seeds (the sign-dependent part's two signs as independent samples), the between-seed "
                 r"spread, and the shared baseline once (3 seeds). Coherent: units without a degenerate reply. Refusal steer \citep{Arditi2024-hz}: "
                 r"$c=\pm0.5$, one run per sign. $^\ddagger$Control ReLU steer on benign prompts: seeds 2--3 (no seed-1 $\alpha=-1$ run).")


def table_geometry_evenodd(data: dict, caption: str = TABLE_CAPTION, label: str = "tab:app_evenodd_signs") -> str:
    """The LaTeX table: for each object D(+), D(-), the sign-independent and the sign-dependent part, per category, all replies and the
    coherent subset; the first row the unsteered level (`unsteered_levels`). These are the sign rows of the appendix's tab:app_evenodd;
    its interaction rows need the constitution and difference objects, which this sidecar does not hold."""
    parts = {m: category_parts(data, coherent=(m == "coherent")).set_index(["object", "category"]) for m in ("all", "coherent")}
    lv = {m: unsteered_levels(data, coherent=(m == "coherent")) for m in ("all", "coherent")}
    cats = list(CATEGORIES)
    dagger = {c: (r"$^\ddagger$" if c == "over-refusal" else "") for c in cats}
    head = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{CATEGORY_LABEL[c]}{dagger[c]}}}" for c in cats) + r" \\"
    rule = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cats)))
    sub = " & " + " & ".join("all & coherent" for _ in cats) + r" \\"
    lines = [r"\begin{table}[tbp]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\small",
             r"\begin{tabular}{@{}l " + " ".join("cc" for _ in cats) + "@{}}", r"\toprule", head, rule, sub, r"\midrule",
             r"\emph{Unsteered (seed average)} & " + " & ".join(f"${lv[m][c]:.1f}$" for c in cats for m in ("all", "coherent")) + r" \\", r"\midrule"]
    names = {"control_relu": r"Control ReLU steer", "control_lora": r"Control LoRA", "arditi_allpos": r"Refusal steer ($c=\pm0.5$)"}
    sign_rows = (("Dp", r"$D(+\theta)$"), ("Dn", r"$D(-\theta)$"), ("even", "sign-independent"), ("odd", "sign-dependent"))
    for i, o in enumerate(OBJECTS):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{1 + 2 * len(cats)}}}{{@{{}}l}}{{\emph{{{names[o]}}}}}\\")
        for q, nm in sign_rows:
            cells = []
            for c in cats:
                for m in ("all", "coherent"):
                    x = parts[m].loc[(o, c)] if (o, c) in parts[m].index else None
                    cells.append(_pm(None if x is None else x[q], None if x is None else x[f"{q}_ci"]))
            lines.append(rf"\quad {nm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def not_drawn(data: dict, coherent: bool = False) -> list:
    """What the figure leaves out, one phrase each: every unpaired cell (one sign only, so neither sign is drawn) and every mean
    `appendix_common` does not draw. Empty when the figure draws everything the sidecar holds."""
    out = [f"{R.LABEL.get(u['object'], u['object'])} {u['seed']} {R.EVAL_LABEL[u['eval']]}: no {'α = −1' if u['missing_sign'] == -1 else 'α = +1'} "
           f"run (the registry holds none for that seed's object), so neither sign is drawn and the seed is out of both means"
           for u in data.get("unpaired") or []]
    return out + AC.not_drawn(rows(data, coherent), CATEGORIES, labels=LABELS, condition_labels=CONDITION_LABELS)


def figure_geometry_evenodd(data: dict, name: str | None = FIGURE, coherent: bool = False, layout=None, out_dirs=None, extra_dirs=(), **kw):
    """The paired comparison (see the module docstring), saved as `name` (png, pdf, svg into `R.FIGS` and the mirrors, or `out_dirs`) when
    one is given, and returned. `coherent` draws the union5-coherent recount; `layout` defaults to static on top, multi-turn and benign
    compliance below; other keywords go to `appendix_common.paired_comparison_figure`."""
    return AC.paired_comparison_figure(rows(data, coherent), name, categories=CATEGORIES, conditions=list(CONDITIONS), objects=list(OBJECTS),
                                       layout=layout or LAYOUT, labels=LABELS, condition_labels=CONDITION_LABELS, titles=TITLES,
                                       out_dirs=out_dirs, extra_dirs=extra_dirs, **kw)
