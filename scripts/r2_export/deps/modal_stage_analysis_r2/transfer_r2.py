#!/usr/bin/env python3
"""sdf-base-steering round 2 — controls-in-battery.

Reads the r2-namespace battery aggregates (same schema as round 1's
``two_si_<attack>_aggregate.json``) and emits, in the round-1
``transfer_stats.json`` schema (``models.<slug>.cells.<cell>.<readout>`` =
Wilson record; ``models.<slug>.vs_bare.<cell>.<readout>`` = paired contrast):

  * the r2-wave rows: same-wave bare anchor (``none``), the two ctl cells,
    the regraded round-1 bare responses (``none_r1``), any extras;
  * every contrast PAIRED WITHIN THE r2 WAVE only (never across waves);
  * ctl-vs-bare reported additionally as EQUIVALENCE BOUNDS (the paired
    95% CI half-width is the measured ceiling on how much register-only
    training can move refusal);
  * the DRIFT INSET: (i) generation drift = r2-bare vs regraded r1-bare, both
    graded in the r2 pass; (ii) judge drift = regraded r1-bare vs r1-bare's
    ORIGINAL round-1 grades (from the round-1 aggregates). Goal-level
    McNemar + paired bootstrap on the shared goals.

The round-1 rows are carried read-only from round 1's transfer_stats.json so
the appended table shows both waves, visibly grouped.

Readers are round 1's own (``transfer_stats.gated_cell`` etc.), imported by
path — never a reimplementation. Every input absent -> PENDING.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from r2_paths import (Layout, MODELS, XFER_ATTACKS, XFER_MODELS,  # noqa: E402
                      XFER_NAMESPACES, XFER_R2_CELLS, XFER_REGRADE_CELL, R1_ANALYSIS, read_json, xfer_object_owner)
import r2_stats as S  # noqa: E402

sys.path.insert(0, str(R1_ANALYSIS))
import transfer_stats as T1  # noqa: E402  (round-1 readers, by path)

GATED = T1.GATED
ATTACKS = T1.ATTACKS
THRESHOLD = float(T1.THRESHOLD)
BON_N = int(T1.BON_N)
READOUTS = [*GATED, "bon", "xstest", "mmlu", "gsm8k"]

CELL_LABEL_R2 = {
    "none": "none (bare instruct, same wave)",
    "constv2ctl-100k__psr_ce_all": "REGISTER CONTROL · PSR (CE, ctl-100k)",
    "constv2ctl-100k__lora_r4_ce": "REGISTER CONTROL · LoRA r4 (CE, ctl-100k)",
    XFER_REGRADE_CELL: "none — round-1 bare RESPONSES regraded in the r2 pass",
    "none_r2": "none — r2 bare RESPONSES regraded in the r3 pass",
    "none_r3": "none — r3 bare RESPONSES regraded in the it pass",
    # the 1M base-fit cells ride along as extras; their OBJECT GENERATION is read from the
    # capability leg's artifact sha (see object_generation), never assumed
    "constv2-mix-1m__psr_ce_all": "BASE-fit PSR (CE, 1M)",
    "constv2-mix-1m__lora_r4_ce": "BASE-fit LoRA r4 (CE, 1M)",
    "prompted__si_constitution": "prompted (SI in system prompt) — re-run",
    "si_constitution__psr_kl_all": "SI-distilled PSR (KL, all pos) — re-run",
    "si_constitution__lora_r4": "SI-distilled LoRA r4 (KL) — re-run",
}
CELL_GROUP_R2 = {
    "none": "baseline_r2",
    "constv2ctl-100k__psr_ce_all": "register_control",
    "constv2ctl-100k__lora_r4_ce": "register_control",
    XFER_REGRADE_CELL: "drift_regrade",
    "none_r2": "drift_regrade",
    "none_r3": "drift_regrade",
    "constv2-mix-1m__psr_ce_all": "base_fit_1m",
    "constv2-mix-1m__lora_r4_ce": "base_fit_1m",
    "prompted__si_constitution": "reference_rerun",
    "si_constitution__psr_kl_all": "reference_rerun",
    "si_constitution__lora_r4": "reference_rerun",
}
#: labels / groups that depend on the NAMESPACE a cell id appears in. In the `it` wave the constv2-mix-full
#: ids are the INSTRUCT-DIRECT objects — trained on the instruct checkpoint under the <base>_it_doc keys, their own
#: generation (not a base→instruct staged object) — and gemma's 1M LoRA row is the re-run on its current cap-60
#: object. Object identity for these rows binds through r2_paths.xfer_object_owner, never through the base key.
CELL_LABEL_NS = {
    "it": {"constv2-mix-full__psr_ce_all": "INSTRUCT-DIRECT PSR (CE, full corpus; trained ON the instruct checkpoint)",
           "constv2-mix-full__lora_r4_ce": "INSTRUCT-DIRECT LoRA r4 (CE, full corpus; trained ON the instruct checkpoint)",
           "constv2-mix-1m__lora_r4_ce": "BASE-fit LoRA r4 (CE, 1M) — re-run on the current object (R2-D66)"},
}
CELL_GROUP_NS = {
    "it": {"constv2-mix-full__psr_ce_all": "instruct_direct", "constv2-mix-full__lora_r4_ce": "instruct_direct"},
}
#: groups whose rows carry an object generation read from the capability leg's artifact sha
OBJECT_GROUPS = ("base_fit_1m", "register_control", "instruct_direct", "extra")


def cell_label(cell: str, ns: str = "r2") -> str:
    return CELL_LABEL_NS.get(ns, {}).get(cell) or CELL_LABEL_R2.get(cell, f"{cell} (extra)")


def cell_group(cell: str, ns: str = "r2") -> str:
    return CELL_GROUP_NS.get(ns, {}).get(cell) or CELL_GROUP_R2.get(cell, "extra")


def _gen_sort_key(label: str):
    """The continuation table's resolved generation first (the current object, then a chain publication by cap), then
    the other generation names (wave-A / round-1 / cap-N snapshot / instruct-direct / derived parent), then the run-label
    and stamp sources that agree with them."""
    if label.startswith("current object"):
        rank = 0
    elif label.startswith("cap-") and " publication (generation " in label:
        rank = 1
    elif label.startswith(("cap-", "wave-A", "round-1", "instruct-direct", "derived_meta")):
        rank = 2
    else:
        rank = 3
    return (rank, label)


def not_in_design(reason: str) -> dict:
    return {"status": "NOT_IN_DESIGN", "reason": reason}


def object_generation(L: Layout, model: str, cell: str, ns: str, known: Optional[dict]) -> dict:
    """Which object generation a battery row served. The aggregate records no
    object identity; the capability leg's controls_summary.json carries
    provenance.artifact_sha256, compared against every known generation of the
    cell (MMLU identities per label, the snapshot banks, derived_meta parents,
    probe stamps, the continuation table's resolved current object). The cell's
    object OWNER is per namespace (r2_paths.xfer_object_owner): the `it` wave's
    constv2-mix-full cells are the instruct-direct objects of the <base>_it_doc
    keys, never the base key's same-named cells. The attack legs of the same wave
    are labelled as such — their identity is not recorded anywhere, so it is
    never asserted."""
    owner = xfer_object_owner(ns, model, cell)
    p = L.transfer_controls_summary(model, cell, ns)
    cs = read_json(p)
    art = ((cs or {}).get("provenance") or {}).get("artifact_sha256")
    ko = ((known or {}).get(owner) or {}).get(cell) or {}
    if not art:
        return pending(f"controls_summary.json absent or without provenance.artifact_sha256 "
                       f"({L.resolved.get(f'{XFER_NAMESPACES[ns]["dir"]}/{XFER_MODELS[model]}/controls_summary/{cell}')})")
    hits = sorted({lab for lab, sha in ko.items() if sha == art}, key=_gen_sort_key)
    attack_legs = "same wave as the controls leg (spawned together); identity not recorded in the aggregate"
    if not ko:
        if owner != model:
            return pending(f"no identity-bearing file staged for the instruct-direct key {owner} (cell {cell}): looked for "
                           f"{L.harvest / 'probe_v2' / owner / (cell + '.jsonl.meta.json')} (probe stamp) and "
                           f"{L.harvest / 'mmlu' / owner}/<label>/mmlu_<label>_{cell}.json (MMLU identity)")
        return {"status": "unknown", "artifact_sha256": art, "object_owner": owner,
                "controls_leg": "no identity-bearing file names an object for this cell — generation unknown",
                "attack_legs": attack_legs}
    if not hits:
        return {"status": "unknown", "artifact_sha256": art, "object_owner": owner,
                "controls_leg": ("artifact sha matches NONE of the known generations of this cell: "
                                 + ", ".join(f"{lab}={sha[:12]}…" for lab, sha in ko.items())),
                "attack_legs": attack_legs}
    return {"status": "ok", "artifact_sha256": art, "object_owner": owner, "controls_leg": " = ".join(hits),
            "attack_legs": attack_legs, "known": ko}


def pending(reason: str) -> dict:
    return {"status": "PENDING", "reason": reason}


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (SystemExit, KeyError, TypeError, ValueError) as e:
        return pending(f"{type(e).__name__}: {e}")


def _wave_cells(aggs: dict, expected=XFER_R2_CELLS, regrade=XFER_REGRADE_CELL) -> list[str]:
    seen: list[str] = []
    for a in ATTACKS:
        agg = aggs.get(a)
        if not agg:
            continue
        for c in agg["cells"]:
            if c not in seen:
                seen.append(c)
    order = [c for c in expected if c in seen]
    order += [c for c in seen if c not in order and c == regrade]
    order += sorted(c for c in seen if c not in order)
    return order


def cell_rows(aggs: dict, cell: str, ns: str = "r2") -> dict:
    row: dict = {"label": cell_label(cell, ns), "group": cell_group(cell, ns)}
    for a in GATED:
        agg = aggs.get(a)
        row[a] = (_safe(T1.gated_cell, agg, cell) if agg and cell in agg["cells"]
                  else pending(f"{a} aggregate absent"))
    agg = aggs.get("bon")
    row["bon"] = (_safe(T1.bon_cell, agg, cell) if agg and cell in agg["cells"]
                  else pending("bon aggregate absent"))
    # controls are attack-independent: read from the first aggregate that has
    # the cell, and cross-check the others exactly as round 1 did
    src = next((aggs[a] for a in ATTACKS if aggs.get(a) and cell in aggs[a]["cells"]),
               None)
    if src is None:
        for k in ("xstest", "mmlu", "gsm8k"):
            row[k] = pending("no aggregate carries this cell")
        return row
    row["xstest"] = _safe(T1.xstest_cell, src, cell)
    row["mmlu"] = _safe(T1.control_cell, src, cell, "mmlu")
    row["gsm8k"] = _safe(T1.control_cell, src, cell, "gsm8k")
    for a in ATTACKS:
        other = aggs.get(a)
        if not other or other is src or cell not in other["cells"]:
            continue
        for key, getter in (("xstest", T1.xstest_cell),
                            ("mmlu", lambda g, c: T1.control_cell(g, c, "mmlu")),
                            ("gsm8k", lambda g, c: T1.control_cell(g, c, "gsm8k"))):
            if row[key].get("status") == "PENDING":
                continue
            o = _safe(getter, other, cell)
            if o.get("status") == "PENDING":
                continue
            if (o["k"], o["n"]) != (row[key]["k"], row[key]["n"]):
                raise SystemExit(
                    f"[transfer_r2] {cell}: control {key} disagrees between "
                    f"aggregates ({row[key]['k']}/{row[key]['n']} vs "
                    f"{o['k']}/{o['n']}) — attack-independent, must be identical")
    return row


# ---------------------------------------------------------------- per-item
def attack_outcomes(agg: Optional[dict], cell: str) -> Optional[dict]:
    if not agg or cell not in agg["cells"]:
        return None
    pg = agg["cells"][cell]["attack_metrics"]["per_goal"]
    return {gid: int(g["best_score"] > THRESHOLD) for gid, g in pg.items()}


def bon_outcomes(agg: Optional[dict], cell: str) -> Optional[dict]:
    if not agg or cell not in agg["cells"]:
        return None
    ns = agg["cells"][cell]["attack_metrics"]["n_star_by_goal"]
    return {gid: int(0 < int(v) <= BON_N) for gid, v in ns.items()}


def controls_outcomes(run_dir: Path, cell: str) -> dict[str, dict]:
    """{task: {item_id: 0|1}} from the cell-controls store; {} if the store
    or the repo's key function is unavailable (contrast then unpaired)."""
    root = run_dir / "controls" / "cells"
    if not root.exists():
        return {}
    try:
        from sisteer.eval.cell_controls import store_key_token  # type: ignore
        tok = store_key_token(cell)
        hits = sorted(p for p in root.iterdir()
                      if p.name.startswith(f"{tok}__controls__"))
    except Exception:  # pragma: no cover - repo env absent
        hits = sorted(p for p in root.iterdir()
                      if p.name.startswith(f"{cell}__controls__"))
    if len(hits) != 1:
        return {}
    out: dict[str, dict] = {}
    for line in (hits[0] / "rows.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("item_id") == "__aggregate__" or r.get("metric") != "acc":
            continue
        out.setdefault(r["task"], {})[r["item_id"]] = int(round(float(r["value"])))
    return out


def xstest_outcomes(priv_dir: Path, cell: str) -> dict:
    """{goal_id: 1 if FULL refusal} from the r2 grade cache; {} if any piece
    is missing (no API calls are ever made here)."""
    try:
        from grade_bon_live import ScoreCache  # type: ignore
        from sisteer.eval.attacks.grading_live import _live_cache_key  # type: ignore
        from sisteer.eval.jailbreak_data import load_xstest_safe  # type: ignore
    except Exception:  # pragma: no cover
        return {}
    caches = sorted(priv_dir.glob("live_two_si_grade_cache*.jsonl"))
    path = priv_dir / cell / "xstest_responses.jsonl"
    if not caches or not path.exists():
        return {}
    cache = ScoreCache(caches[0], judge_model="claude-haiku-4-5")
    xs = {g.goal_id: g for g in load_xstest_safe(revision=T1.XSTEST_REV)}
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = xs.get(r["goal_id"])
        if g is None:
            continue
        v = cache.get(_live_cache_key("xs", g.dataset, g.revision, g.text,
                                      r["response"]))
        if v is None:
            return {}
        out[r["goal_id"]] = 1 if v["label"] == "I" else 0
    return out


def per_item(aggs: dict, cells: list[str], run_dir: Optional[Path],
             priv_dir: Path) -> dict:
    per: dict[str, dict] = {}
    for a in GATED:
        per[a] = {c: attack_outcomes(aggs.get(a), c) for c in cells}
    per["bon"] = {c: bon_outcomes(aggs.get("bon"), c) for c in cells}
    if run_dir is not None:
        ctl = {c: controls_outcomes(run_dir, c) for c in cells}
        for key, task in T1.CTRL_TASK.items():
            per[key] = {c: (ctl[c].get(task) or None) for c in cells}
    xs = {c: xstest_outcomes(priv_dir, c) for c in cells}
    per["xstest"] = {c: (xs[c] or None) for c in cells}
    return per


def strict_contrast(a: dict, b: dict, names=("bare", "cell")) -> dict:
    """Round 1's paired_contrast, but only on EXACTLY the same item ids —
    a missing/extra/duplicate id is REFUSED (with the ids), never intersected."""
    try:
        S.check_same_ids(a, b, names)
    except S.PairingError as e:
        return e.as_dict()
    return T1.paired_contrast(a, b)


def contrast(row_a: dict, row_b: dict, pa: Optional[dict], pb: Optional[dict]) -> dict:
    """cell(b) minus bare(a) for one readout — the round-1 record shape."""
    if row_a.get("status") in ("PENDING", "REFUSED") or row_b.get("status") in ("PENDING", "REFUSED"):
        return pending("readout pending on one side")
    rec = {"delta_pp": 100 * (row_b["p"] - row_a["p"]),
           "cell": {"k": row_b["k"], "n": row_b["n"], "p": row_b["p"]},
           "bare": {"k": row_a["k"], "n": row_a["n"], "p": row_a["p"]},
           "ci_overlap": not (row_b["hi"] < row_a["lo"] or row_a["hi"] < row_b["lo"])}
    if pa and pb:
        p = strict_contrast(pa, pb)
        if p.get("status") == "REFUSED":
            rec["paired_refused"] = p
        else:
            rec["paired"] = p
            rec["equivalence_bound_pp"] = max(abs(p["lo_pp"]), abs(p["hi_pp"]))
    return rec


# ---------------------------------------------------------------- drift manifests
def _manifest_rows(block: dict, name: str) -> list[tuple[str, str]]:
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        raise S.PairingError(f"drift manifest {name}: rows missing/empty")
    out = []
    for r in rows:
        if "goal_id" not in r or "response_sha256" not in r:
            raise S.PairingError(f"drift manifest {name}: a row lacks goal_id/response_sha256")
        out.append((str(r["goal_id"]), str(r["response_sha256"])))
    d = S._dups([g for g, _ in out])
    if d:
        raise S.PairingError(f"drift manifest {name}: duplicate goal ids", duplicates=d)
    return out


def validate_drift_manifest(m: Optional[dict], attack: str, goals_r2: dict, goals_r1: Optional[set],
                            regrade: str = XFER_REGRADE_CELL) -> dict:
    """``goals_r2[cell]`` = the goal ids THIS
    wave's aggregate graded for that cell; ``goals_r1`` = the goal ids of the
    regraded cell's ORIGINAL grading (None if that aggregate is absent);
    ``regrade`` = the regrade cell's name (none_r1 in the r2 wave, none_r2 in
    the r3 wave; the manifest names <regrade> and <regrade>_original).
    Returns {"judge_leg": ok|REFUSED|PENDING, "gen_leg": ...} with reasons."""
    def refused(reason, **kw):
        return S.PairingError(reason, **kw).as_dict()

    if m is None:
        return {"judge_leg": pending("drift manifest absent"), "gen_leg": pending("drift manifest absent")}
    if m.get("attack") != attack:
        r = refused(f"manifest attack {m.get('attack')!r} != {attack!r}")
        return {"judge_leg": r, "gen_leg": r}
    cells = m.get("cells") or {}
    judge = m.get("judge") or {}
    orig = f"{regrade}_original"
    out: dict = {}
    # ---- judge-drift leg: byte-identical responses between original and regraded rows
    try:
        if regrade not in cells or orig not in cells:
            out["judge_leg"] = pending(f"manifest lacks {regrade} and/or {orig}")
        else:
            a = _manifest_rows(cells[regrade], regrade)
            b = _manifest_rows(cells[orig], orig)
            da, db = dict(a), dict(b)
            if set(da) != set(db):
                raise S.PairingError("regraded and original round-1 rows do not carry the same goals",
                                     missing=set(da) - set(db), extra=set(db) - set(da))
            diff = sorted(g for g in da if da[g] != db[g])
            if diff:
                raise S.PairingError("regraded round-1 responses are NOT byte-identical to the original "
                                     "round-1 responses (response_sha256 differs)", extra=diff)
            if a != b:
                raise S.PairingError("regraded and original round-1 rows differ in ORDER (manifests must be "
                                     "ordered identically)")
            if regrade in goals_r2 and set(da) != goals_r2[regrade]:
                raise S.PairingError(f"manifest {regrade} goals != goals graded in this wave's aggregate",
                                     missing=set(da) - goals_r2[regrade],
                                     extra=goals_r2[regrade] - set(da))
            if goals_r1 is not None and set(db) != goals_r1:
                raise S.PairingError(f"manifest {orig} goals != goals in the original grading's aggregate",
                                     missing=set(db) - goals_r1, extra=goals_r1 - set(db))
            out["judge_leg"] = {"status": "ok", "n_goals": len(a),
                                "original_pass_id": cells[orig].get("pass_id"),
                                "original_judge": cells[orig].get("judge")}
    except S.PairingError as e:
        out["judge_leg"] = e.as_dict()
    # ---- generation-drift leg: same judge pass for this wave's bare and the regraded bare
    try:
        if "none" not in cells or regrade not in cells:
            out["gen_leg"] = pending(f"manifest lacks none and/or {regrade}")
        else:
            pid = judge.get("pass_id")
            p0, p1 = cells["none"].get("pass_id"), cells[regrade].get("pass_id")
            if not pid or not judge.get("model") or not judge.get("rubric"):
                raise S.PairingError("manifest judge provenance incomplete (model/rubric/pass_id required)")
            if not (p0 == p1 == pid):
                raise S.PairingError(f"this wave's bare and the regraded {regrade} were not graded in the same pass "
                                     f"(pass ids {p0!r}, {p1!r}; judge.pass_id {pid!r})")
            a = dict(_manifest_rows(cells["none"], "none"))
            b = dict(_manifest_rows(cells[regrade], regrade))
            if set(a) != set(b):
                raise S.PairingError(f"this wave's bare and the regraded {regrade} rows do not carry the same goals",
                                     missing=set(a) - set(b), extra=set(b) - set(a))
            for c, rows in (("none", a), (regrade, b)):
                if c in goals_r2 and set(rows) != goals_r2[c]:
                    raise S.PairingError(f"manifest {c} goals != goals graded in this wave's aggregate",
                                         missing=set(rows) - goals_r2[c], extra=goals_r2[c] - set(rows))
            out["gen_leg"] = {"status": "ok", "n_goals": len(a), "pass_id": pid,
                              "judge": {"model": judge["model"], "rubric": judge["rubric"]}}
    except S.PairingError as e:
        out["gen_leg"] = e.as_dict()
    return out


# ---------------------------------------------------------------- build
def build(L: Layout, known_objects: Optional[dict] = None) -> dict:
    r1 = read_json(L.r1_transfer)
    out: dict = {
        "meta": {
            "threshold": THRESHOLD, "bon_n_max": BON_N,
            "interval": "Wilson score, 95%",
            "pairing": "10,000-draw percentile bootstrap over items, seed 0; "
                       "exact McNemar on discordant pairs; WITHIN-WAVE only",
            "r2_cells": XFER_R2_CELLS, "regrade_cell": XFER_REGRADE_CELL,
            "namespaces": {ns: {k: v for k, v in cfg.items() if k != "cells"} | {"cells": cfg["cells"]}
                           for ns, cfg in XFER_NAMESPACES.items()},
            "cell_labels": CELL_LABEL_R2, "cell_groups": CELL_GROUP_R2,
            "cell_labels_by_namespace": CELL_LABEL_NS, "cell_groups_by_namespace": CELL_GROUP_NS,
            "object_owner_rule": ("a battery cell's object identity binds to r2_paths.xfer_object_owner(namespace, model, "
                                  "cell): the `it` wave's constv2-mix-full cells are the instruct-direct objects of the "
                                  "<base>_it_doc keys (R2-D50/D64), every other cell the served base key's"),
            "r1_source": str(L.r1_transfer) if r1 else None,
            "object_generation_rule": ("a battery row's object generation is read from the capability leg's "
                                       "controls_summary.json provenance.artifact_sha256 against the known "
                                       "generations of the cell; the attack legs share the wave but record no "
                                       "identity, so they are labelled 'same wave', never asserted"),
        },
        "r1": ({m: {"label": v["label"], "cells": v["cells"], "vs_bare": v["vs_bare"]}
                for m, v in r1["models"].items()} if r1 else None),
        "namespaces": {}, "models": {}, "drift": {},
    }
    if r1:
        out["meta"]["r1_cells"] = r1["meta"]["cells"]
        out["meta"]["r1_cell_labels"] = r1["meta"]["cell_labels"]
        out["meta"]["gated_goal_n"] = r1["meta"].get("gated_goal_n")

    for ns, cfg in XFER_NAMESPACES.items():
        block: dict = {"label": cfg["label"], "regrade": cfg["regrade"], "models": {}, "drift": {}}
        out["namespaces"][ns] = block
        for model in MODELS:
            _build_namespace_model(L, ns, cfg, model, r1, block, known_objects)
    # the names the rest of the pipeline grew up with = the r2 namespace
    out["models"] = out["namespaces"]["r2"]["models"]
    out["drift"] = out["namespaces"]["r2"]["drift"]
    return out


def _build_namespace_model(L: Layout, ns: str, cfg: dict, model: str, r1, block: dict,
                           known_objects: Optional[dict]) -> None:
    slug = XFER_MODELS[model]
    sub = cfg["dir"]
    aggs = {a: read_json(L.transfer_ns_aggregate(model, a, ns)) for a in ATTACKS}
    present = [a for a in ATTACKS if aggs[a]]
    expected = list(cfg["cells"].get(model, []))
    if not present:
        staged_manifests = [a for a in ATTACKS if L.drift_manifest(model, a, ns) is not None]
        block["models"][slug] = pending(
            f"no {ns} battery aggregates staged for namespace sdfbase_transfer_{slug}{cfg['suffix']} "
            f"(looked in {L.resolved.get(f'{sub}/{slug}')}; battery rows are read from the harvest root only, "
            f"never from the battery's repo out dir); expected cells: {expected}"
            + (f"; {len(staged_manifests)} drift manifest(s) already staged ({', '.join(staged_manifests)})" if staged_manifests else ""))
        block["drift"][slug] = (pending(f"{ns} aggregates not staged"
                                        + (f" — {len(staged_manifests)} drift manifests staged ({', '.join(staged_manifests)}): the "
                                           f"regraded {cfg['regrade']} rows are in place; the generation leg needs this wave's bare "
                                           "generations and the aggregates" if staged_manifests else ""))
                                if cfg["regrade"] else not_in_design("no regrade cell in this namespace's design"))
        return
    cells = _wave_cells(aggs, expected, cfg["regrade"] or XFER_REGRADE_CELL)
    rows = {c: cell_rows(aggs, c, ns) for c in cells}
    for c in cells:
        grp = cell_group(c, ns)
        if grp in OBJECT_GROUPS:
            rows[c]["object_generation"] = object_generation(L, model, c, ns, known_objects)
        elif grp == "reference_rerun":
            rows[c]["object_generation"] = {"status": "reference",
                                            "controls_leg": ("round-1 reference (prompted: no object; SI-distilled: the "
                                                             "si-steering campaign's object, identity not tracked by "
                                                             "this analysis) re-run in this wave")}
    first = aggs[present[0]]
    m: dict = {"label": (r1["models"][slug]["label"] if r1 else slug), "namespace": ns,
               "attacks_present": present,
               "attacks_pending": [a for a in ATTACKS if a not in present],
               "expected_cells": expected,
               "expected_missing": {c: pending(f"expected cell `{c}` absent from the {ns} aggregates")
                                    for c in expected if c not in cells},
               "served_alpha": first.get("served_alpha"),
               "cells": rows, "vs_bare": {}, "equivalence_bounds": {}}
    if "none" not in cells:
        m["vs_bare"] = pending("same-wave bare anchor `none` absent — no within-wave contrast possible")
    else:
        per = per_item(aggs, cells, L.transfer_ns_dir(model, ns), L.transfer_ns_private(model, ns))
        for c in cells:
            if c == "none":
                continue
            m["vs_bare"][c] = {}
            for key in READOUTS:
                pa = per.get(key, {}).get("none")
                pb = per.get(key, {}).get(c)
                m["vs_bare"][c][key] = contrast(rows["none"][key], rows[c][key], pa, pb)
            if cell_group(c, ns) == "register_control":
                m["equivalence_bounds"][c] = {
                    key: (m["vs_bare"][c][key].get("equivalence_bound_pp")
                          if "paired" in m["vs_bare"][c][key] else None)
                    for key in READOUTS}
    block["models"][slug] = m
    if not cfg["regrade"]:
        block["drift"][slug] = not_in_design("no regrade cell in this namespace's design")
        return
    _drift_inset(L, ns, model, slug, aggs, cells, block, cfg["regrade"], cfg.get("original_from"))


def _original_aggregate(L: Layout, model: str, attack: str, original_from: Optional[str]):
    """The aggregate holding the regraded cell's ORIGINAL grades: round 1's (repo out
    dir) for the r2 wave's none_r1; the r2 harvest aggregate for the r3 wave's none_r2."""
    if original_from == "r1_repo_out":
        return read_json(L.transfer_r1_aggregate(model, attack))
    if original_from in XFER_NAMESPACES:
        return read_json(L.transfer_ns_aggregate(model, attack, original_from))
    return None


def _drift_inset(L: Layout, ns: str, model: str, slug: str, aggs: dict, cells: list, block: dict,
                 regrade: str, original_from: Optional[str]) -> None:
    drift: dict = {}
    if True:
        if regrade not in cells or "none" not in cells:
            drift = pending(f"regrade cell `{regrade}` and/or `none` "
                            f"absent from the {ns} aggregates")
        else:
            for a in ATTACKS:
                agg2 = aggs.get(a)
                if not agg2:
                    drift[a] = pending(f"{a} {ns} aggregate absent")
                    continue
                get = bon_outcomes if a == "bon" else attack_outcomes
                r2_bare = get(agg2, "none")
                r1_regraded = get(agg2, regrade)
                agg1 = _original_aggregate(L, model, a, original_from)
                r1_orig = get(agg1, "none") if agg1 else None
                man = read_json(L.drift_manifest(model, a, ns))
                checks = validate_drift_manifest(
                    man, a,
                    {c: set(o) for c, o in (("none", r2_bare), (regrade, r1_regraded)) if o},
                    set(r1_orig) if r1_orig else None, regrade=regrade)
                gen = judge = None
                if checks["gen_leg"].get("status") == "ok" and r2_bare and r1_regraded:
                    gen = strict_contrast(r1_regraded, r2_bare, (f"regraded {regrade}", f"{ns} bare"))
                if checks["judge_leg"].get("status") == "ok" and r1_orig and r1_regraded:
                    judge = strict_contrast(r1_orig, r1_regraded, (f"{regrade} original grades", f"{regrade} regraded"))
                drift[a] = {
                    "manifest": L.resolved.get(f"drift_manifest/{ns}/{slug}/{a}"),
                    "manifest_checks": checks,
                    "generation_drift": (gen if gen else
                                         checks["gen_leg"] if checks["gen_leg"].get("status") != "ok" else
                                         pending(f"none/{regrade} per-goal outcomes missing")),
                    "judge_drift": (judge if judge else
                                    checks["judge_leg"] if checks["judge_leg"].get("status") != "ok" else
                                    pending(f"the aggregate holding {regrade}'s original grades ({original_from}) not found")),
                    "regrade_cell": regrade,
                    "read": f"generation_drift = {ns}-bare minus regraded {regrade} "
                            f"(both graded in the {ns} pass); judge_drift = "
                            f"regraded {regrade} minus its original grades ({original_from})",
                }
        block["drift"][slug] = drift


# ---------------------------------------------------------------- markdown
COLS = T1.COLS


def _fmt(rec):
    if rec is None or rec.get("status") == "PENDING":
        return "PENDING"
    return T1.fmt_pct(rec)


def _fmt_contrast(v):
    if v is None or v.get("status") in ("PENDING", "REFUSED"):
        return v.get("status", "PENDING") if v else "PENDING"
    p = v.get("paired")
    if not p:
        if v.get("paired_refused"):
            return f"{v['delta_pp']:+.1f} (pairing REFUSED)"
        return f"{v['delta_pp']:+.1f} (unpaired)"
    star = "" if p["excludes_zero"] else " n.s."
    return f"{p['delta_pp']:+.1f} [{p['lo_pp']:+.1f}, {p['hi_pp']:+.1f}]{star}"


def _gen_label(row: dict) -> str:
    og = row.get("object_generation")
    if not og:
        return ""
    if og.get("status") == "ok":
        return f" [object: {og['controls_leg']} — controls-leg sha verified; attack legs same wave]"
    if og.get("status") == "unknown":
        return " [object generation UNKNOWN: " + og["controls_leg"][:80] + "]"
    if og.get("status") == "reference":
        return " [" + og["controls_leg"] + "]"
    return " [object generation PENDING]"


def markdown(X: dict) -> str:
    out: list[str] = []
    head = "| wave | cell | " + " | ".join(h for _, h in COLS) + " |"
    rule = "|---" * (len(COLS) + 2) + "|"
    for model in MODELS:
        slug = XFER_MODELS[model]
        m = X["models"].get(slug)
        label = (m or {}).get("label") or (X["r1"] or {}).get(slug, {}).get("label", slug)
        out.append(f"### {label}  (`{slug}`) — battery, every wave\n")
        out.append(head)
        out.append(rule)
        if X["r1"] and slug in X["r1"]:
            for cell in X["meta"]["r1_cells"]:
                row = X["r1"][slug]["cells"][cell]
                out.append(f"| r1 | {row['label']} | "
                           + " | ".join(_fmt(row[k]) for k, _ in COLS) + " |")
        for ns, block in X["namespaces"].items():
            mm = block["models"].get(slug)
            if mm is None or mm.get("status") == "PENDING":
                out.append(f"| {ns} | PENDING — {(mm or {}).get('reason', 'not staged')} | "
                           + " | ".join("PENDING" for _ in COLS) + " |")
                continue
            for cell, row in mm["cells"].items():
                out.append(f"| **{ns}** | {row['label']}{_gen_label(row)} | "
                           + " | ".join(_fmt(row[k]) for k, _ in COLS) + " |")
            for cell, pend in (mm.get("expected_missing") or {}).items():
                out.append(f"| **{ns}** | {cell_label(cell, ns)} — {pend['reason']} | "
                           + " | ".join("PENDING" for _ in COLS) + " |")
        out.append("")
        out.append("Round-1 rows are carried READ-ONLY from round 1's transfer_stats.json; r2 rows come from the "
                   "fresh `sdfbase_transfer_<model>_r2` namespace with its own same-wave bare anchor, r3 rows from "
                   "`sdfbase_transfer_<model>_r3` (fresh bare anchor + BOTH current 1M arm cells per model, R2-D32, + the r2 "
                   "bare responses regraded in the r3 pass as `none_r2`, R2-D33), it rows from `sdfbase_transfer_<model>_it` "
                   "(R2-D50/D64/D66: fresh bare anchor + the INSTRUCT-DIRECT full-corpus objects — trained ON the instruct "
                   "checkpoint under the `<model>_it_doc` keys, their own generation, not a base→instruct staged object — + "
                   "gemma's 1M LoRA re-run on its current cap-60 object + the r3 bare responses regraded in the it pass as "
                   "`none_r3`). "
                   "Contrasts are never drawn across waves (greedy generations differ run-to-run from numeric noise "
                   "alone). Every interval is a 95% Wilson score interval on the integer numerator; BoN‡ = best-of-"
                   f"{X['meta']['bon_n_max']} on the 100-goal JBB subset only. `[object: …]` names the object "
                   "generation a row served, read from the capability leg's artifact sha (the aggregate itself records "
                   "no object identity; the attack legs of the same wave are labelled, never asserted). The round-1 "
                   "BASE-fit 1M rows are round-1 (cap-12) objects by construction; the r2 wave's 1M rows served those same "
                   "round-1 objects (R2-D32), so NO current 1M object has battery rows until the r3 wave lands.")
        out.append("")
        for ns, block in X["namespaces"].items():
            mm = block["models"].get(slug)
            if not (mm and mm.get("status") != "PENDING" and isinstance(mm["vs_bare"], dict) and mm["vs_bare"]):
                continue
            out.append(f"**Paired contrasts vs the {ns} bare anchor — {label}** "
                       "(pp; 10,000-draw percentile bootstrap over the shared "
                       "items, seed 0; exact McNemar p in transfer_stats_r2.json)\n")
            out.append("| cell | " + " | ".join(h for _, h in COLS) + " |")
            out.append("|---" * (len(COLS) + 1) + "|")
            for cell, v in mm["vs_bare"].items():
                out.append(f"| {mm['cells'][cell]['label']}{_gen_label(mm['cells'][cell])} | "
                           + " | ".join(_fmt_contrast(v[k]) for k, _ in COLS) + " |")
            out.append("")
        m = X["namespaces"]["r2"]["models"].get(slug)
        if m and m.get("status") != "PENDING" and isinstance(m["vs_bare"], dict) and m["vs_bare"]:
            if m["equivalence_bounds"]:
                out.append("**Equivalence bounds for the register controls** "
                           "(|paired 95% CI| max, pp — the measured ceiling on "
                           "how much register-only training can move each "
                           "readout; the expected result is a null, reported "
                           "as a bound, not as 'no effect found'):\n")
                out.append("| cell | " + " | ".join(h for _, h in COLS) + " |")
                out.append("|---" * (len(COLS) + 1) + "|")
                for cell, eb in m["equivalence_bounds"].items():
                    out.append(f"| {m['cells'][cell]['label']} | " + " | ".join(
                        (f"≤ {eb[k]:.1f}" if eb.get(k) is not None else "unpaired")
                        for k, _ in COLS) + " |")
                out.append("")
        for ns, block in X["namespaces"].items():
            d = block["drift"].get(slug)
            reg = XFER_NAMESPACES[ns].get("regrade") or "—"
            src = XFER_NAMESPACES[ns].get("original_from")
            prev = {"r1_repo_out": "r1"}.get(src, src or "—")          # the wave whose bare responses were regraded
            out.append(f"**Drift inset — {label}, {ns} namespace** (goal-level, paired, pp; regrade cell `{reg}`)\n")
            if not d or d.get("status") in ("PENDING", "NOT_IN_DESIGN"):
                out.append(f"{(d or {}).get('status', 'PENDING')} — {(d or {}).get('reason', 'not staged')}\n")
                continue
            out.append(f"| family | generation drift ({ns} bare − regraded {prev} bare) | "
                       f"judge drift (regraded {prev} − original {prev} grades) |")
            out.append("|---|---|---|")
            for a in ATTACKS:
                v = d.get(a)
                if not v or v.get("status") == "PENDING":
                    out.append(f"| {a} | PENDING | PENDING |")
                    continue
                cells = []
                for key in ("generation_drift", "judge_drift"):
                    p = v[key]
                    if p.get("status") in ("PENDING", "REFUSED"):
                        cells.append(f"{p['status']} — {p.get('reason', '')[:90]}")
                    else:
                        star = "" if p["excludes_zero"] else " n.s."
                        cells.append(f"{p['delta_pp']:+.1f} [{p['lo_pp']:+.1f}, "
                                     f"{p['hi_pp']:+.1f}]{star} "
                                     f"(b/c {p['discordant_bare0_cell1']}/"
                                     f"{p['discordant_bare1_cell0']}, "
                                     f"p={p['mcnemar_exact_p']:.3g})")
                out.append(f"| {a} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, default=None)
    a = ap.parse_args()
    L = Layout(**({"harvest": a.harvest} if a.harvest else {}))
    X = build(L)
    print(markdown(X))
