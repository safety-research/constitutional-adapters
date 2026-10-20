"""`registry degeneracy-rebuild`: regenerate the 2026-09-14 degeneration tables from the registry and diff them.

The 2026-09-14 artifact (`degen_{all,paper,glm}_over1pct.csv`) lists, per pooled cell,
the share of draws / replies flagged by the union of the four text detectors — classifier (tail / line loop),
4-gram repetition > 0.5, zlib collapse, contentless — with AT-CAP EXCLUDED ("at-cap and truncated are not
degeneration"), for every cell above 1 %. The registry stores the five-detector union (`union5`, at-cap included)
with the producer's `by_reason` breakdown in `run_degeneration.extra_json`, so the artifact's statistic is
re-derived here as the sum of the `by_reason` counts whose detector set is not `{at_cap}` alone, over the same
denominator. Sampled cells are pooled over their trials (the artifact's `t*` rows) by summing the per-trial runs.

Output: one CSV in the artifact's 16 columns per input table (`degen_all_over1pct.csv`, and the GLM / paper subsets
by the model's paper_status and the run's status ruling: a descoped model → GLM, a superseded run → all but not paper), and a JSON diff per CSV keyed on
`(view, object_arm, eval_or_attack, cap_or_rung, measure)`: rows only in the CSV (grouped by their source lane — the
ingested lanes are expected here), rows only in the registry, and matched rows whose rate / numerator /
denominator differ. The diff is REPORTED, never forced to zero.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CSV_COLUMNS = ["model", "view", "object_arm", "eval_or_attack", "eval_family", "cap_or_rung", "degen_rate_pct", "numerator", "denominator",
               "denominator_unit", "classifier", "sampler", "lane", "source_file", "measure", "notes"]
KEY = ["view", "object_arm", "eval_or_attack", "cap_or_rung", "measure"]
MODEL_DISPLAY = {"gemma4e4b_it": "Gemma-4-E4B-it", "qwen35_4b_it": "Qwen3.5-4B", "nemotron3nano4b": "Nemotron-3-Nano-4B",
                 "llama8b_it": "Llama-3.1-8B-Instruct", "glm46vflash": "GLM-4.6V-Flash"}
EVAL_DISPLAY = {"pasttense": "past-tense", "codeattack": "CodeAttack", "renellm": "ReNeLLM", "grok420": "grok420", "crescendo": "Crescendo",
                "pair": "PAIR", "actorattack": "ActorAttack", "petri_pooled": "Petri", "strongreject_bon": "best-of-75"}
CSV_DIR_DEFAULT = os.path.join(os.path.expanduser("~"), "figs", "degeneracy", "2026-09-14")
UNION_EXCL_CAP = "union: classify(tail/line) OR 4gram-rep>0.5 OR zlib<0.15 OR contentless (at-cap excluded)"
UNION_EXCL_CAP_SAMPLED = "union: classify OR 4gram-rep>0.5 OR zlib<0.15 OR contentless (at-cap excluded)"


def _j(text) -> Dict[str, Any]:
    try:
        v = json.loads(text) if isinstance(text, str) else (text or {})
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def union_excluding_cap(by_reason: Mapping[str, int]) -> int:
    """Rows flagged by at least one detector other than at-cap (the artifact's estimand) from the union's by_reason counts."""
    n = 0
    for combo, count in (by_reason or {}).items():
        dets = set(str(combo).split("+"))
        if dets - {"at_cap", "at_cap_retokenized", "at_cap_finish_reason"}:
            n += int(count)
    return n


def registry_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """The artifact's rows from the registry: one per pooled (view, cell, eval, instrument) with a measured union5 row."""
    conn.row_factory = sqlite3.Row
    q = """
    SELECT d.run_id, d.numerator, d.denominator, d.denominator_unit, d.extra_json, d.source_uri, d.pooling_json, d.computed_utc,
           w.classifier__all_num, w.rep_ratio_loop__all_num, w.zlib_collapse__all_num, w.contentless__all_num,
           w.at_token_cap_retokenized__all_num, w.at_token_cap_retokenized__all_rate,
           r.view_legacy, r.campaign_cell_id_legacy, r.eval_id, r.instrument_id, r.token_cap_tokens, r.model_id, r.decode_source, r.extra_json AS run_extra,
           r.lane_source_legacy, r.status_ruling_id, i.read_at, m.paper_status
    FROM run_degeneration d
    JOIN v_run_degeneration_wide w ON w.run_id = d.run_id AND w.judgment_id = d.judgment_id
    JOIN run r ON r.run_id = d.run_id
    JOIN instrument i ON i.instrument_id = r.instrument_id
    JOIN model m ON m.model_id = r.model_id
    WHERE d.measure = 'union5' AND d.scope = 'all' AND d.measurement_state = 'measured'
    """
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in conn.execute(q).fetchall():
        row = dict(row)
        extra = _j(row["extra_json"])
        by_reason = extra.get("by_reason") or {}
        n_excl = union_excluding_cap(by_reason)
        den = int(row["denominator"] or 0)
        pooling = _j(row["pooling_json"])
        key = (row["view_legacy"], row["campaign_cell_id_legacy"], row["eval_id"], row["instrument_id"])
        g = groups.setdefault(key, {"num": 0, "den": 0, "classify": 0, "rep": 0, "zlib": 0, "contentless": 0, "cap": 0, "trials": 0,
                                    "row": row, "pooling": pooling, "sources": []})
        g["num"] += n_excl
        g["den"] += den
        g["classify"] += int(row["classifier__all_num"] or 0)
        g["rep"] += int(row["rep_ratio_loop__all_num"] or 0)
        g["zlib"] += int(row["zlib_collapse__all_num"] or 0)
        g["contentless"] += int(row["contentless__all_num"] or 0)
        g["cap"] += int(row["at_token_cap_retokenized__all_num"] or 0)
        g["trials"] += 1
        g["sources"].append(row["source_uri"])
    out: List[Dict[str, Any]] = []
    for (view, cell, eval_id, _inst), g in sorted(groups.items()):
        row = g["row"]
        if not g["den"]:
            continue
        cap = row["token_cap_tokens"]
        ladder = row["read_at"] == "ladder_final"
        sampled = ladder and row["decode_source"] != "greedy"
        unit = str(row["denominator_unit"] or "")
        if sampled:
            unit_txt = "draws (t0+t1 pooled)" if g["trials"] > 1 else "draws (1 trial)"
            origin_label = "sampled static re-run (full22, vendor preset)"
            classifier = UNION_EXCL_CAP_SAMPLED
            cap_txt = f"rung {cap}"
            notes = f"detectors (pooled): classify {g['classify']}, 4gram-rep {g['rep']}, zlib {g['zlib']}, contentless {g['contentless']}; {g['trials']} trial(s)"
            src = g["sources"][0].split(" ›")[0] + " cells[" + "|".join([view, eval_id, cell]) + "|t*].degenerate_share.by_reason" if g["sources"] else ""
        else:
            unit_txt = "target replies (all turns)" if unit == "replies" else "draws (goal x variant, 1 per row)"
            origin_label = f"full22 greedy {'multi-turn' if unit == 'replies' else 'static'} ({cap})"
            classifier = UNION_EXCL_CAP
            cap_txt = str(cap)
            loop_only = 100.0 * g["rep"] / g["den"] if g["den"] else 0.0
            cap_pct = 100.0 * g["cap"] / g["den"] if g["den"] else 0.0
            notes = f"detectors: classify {g['classify']}, 4gram-rep {g['rep']}, zlib {g['zlib']}, contentless {g['contentless']}; loop-only share {loop_only:.2f}%; at-cap {cap_pct:.1f}%"
            src = (g["sources"][0].replace(" ›cells[", " cells[") + ".degenerate_share.by_reason") if g["sources"] else ""
        rextra = _j(row["run_extra"])
        out.append({
            "model": MODEL_DISPLAY.get(row["model_id"], row["model_id"]), "view": view, "object_arm": cell,
            "eval_or_attack": EVAL_DISPLAY.get(eval_id, eval_id), "eval_family": EVAL_DISPLAY.get(eval_id, eval_id), "cap_or_rung": cap_txt,
            "degen_rate_pct": round(100.0 * g["num"] / g["den"], 3), "numerator": g["num"], "denominator": g["den"], "denominator_unit": unit_txt,
            "classifier": classifier, "sampler": rextra.get("decode_declared_legacy") or ("T=0 greedy (no top_p/top_k/min_p/presence); cap %s [registry: instrument decode_source greedy]" % cap
                                                                                        if row["decode_source"] == "greedy" else f"registry: instrument decode_source {row['decode_source']}; cap {cap}"),
            "lane": origin_label, "source_file": src, "measure": "all", "notes": notes, "_paper_status": _paper_status(row), "_model_id": row["model_id"],
        })
    return out


def _paper_status(row: Mapping[str, Any]) -> str:
    """The row's table: `descoped` (the model, or a descope ruling on the run) → the GLM table only; `superseded` (a supersede
    ruling on the run: the greedy 4,096 static and 512 multi-turn instruments) → `all` but not `paper`; else the
    model's paper_status (`in_paper` → the paper table)."""
    ruling = str(row.get("status_ruling_id") or "")
    if row.get("paper_status") == "descoped" or ":descope:" in ruling:
        return "descoped"
    if ":supersede:" in ruling:
        return "superseded"
    return str(row.get("paper_status") or "unverified")


def over_1pct(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows if float(r["degen_rate_pct"]) > 1.0]


def write_tables(rows: Sequence[Mapping[str, Any]], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tables = {
        "degen_all_over1pct.csv": [r for r in rows if r.get("_paper_status") != "descoped"],
        "degen_glm_over1pct.csv": [r for r in rows if r.get("_paper_status") == "descoped"],
        "degen_paper_over1pct.csv": [r for r in rows if r.get("_paper_status") == "in_paper"],
    }
    paths = {}
    for name, rs in tables.items():
        p = os.path.join(out_dir, name)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rs, key=lambda r: -float(r["degen_rate_pct"])):
                w.writerow(r)
        paths[name] = p
    return paths


def _key(r: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(r.get(k, "")) for k in KEY)


def diff_against_csv(rows: Sequence[Mapping[str, Any]], csv_path: str, tol_pct: float = 0.0015) -> Dict[str, Any]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        ref = list(csv.DictReader(fh))
    mine = {_key(r): r for r in rows}
    theirs = {_key(r): r for r in ref}
    only_ref = sorted(set(theirs) - set(mine))
    only_mine = sorted(set(mine) - set(theirs))
    by_origin: Dict[str, int] = {}
    for k in only_ref:
        origin = theirs[k].get("lane") or theirs[k].get("source_file", "")[:60]
        by_origin[origin] = by_origin.get(origin, 0) + 1
    matched = sorted(set(mine) & set(theirs))
    mism = []
    for k in matched:
        a, b = mine[k], theirs[k]
        try:
            d_rate = abs(float(a["degen_rate_pct"]) - float(b["degen_rate_pct"]))
        except (TypeError, ValueError):
            d_rate = float("nan")
        d_num = str(a["numerator"]) != str(b["numerator"])
        d_den = str(a["denominator"]) != str(b["denominator"])
        if not (d_rate <= tol_pct) or d_num or d_den:
            mism.append({"key": "|".join(k), "registry": {"rate": a["degen_rate_pct"], "num": a["numerator"], "den": a["denominator"]},
                         "csv": {"rate": b["degen_rate_pct"], "num": b["numerator"], "den": b["denominator"]}})
    return {"csv": csv_path, "csv_rows": len(ref), "registry_rows": len(rows), "matched": len(matched), "matched_equal": len(matched) - len(mism),
            "mismatched": mism[:50], "n_mismatched": len(mism), "only_in_csv": len(only_ref), "only_in_csv_by_origin": by_origin,
            "only_in_csv_examples": ["|".join(k) for k in only_ref[:10]], "only_in_registry": len(only_mine),
            "only_in_registry_examples": ["|".join(k) for k in only_mine[:10]]}


def rebuild(conn: sqlite3.Connection, out_dir: str, csv_dir: Optional[str] = None) -> Dict[str, Any]:
    rows = over_1pct(registry_rows(conn))
    paths = write_tables(rows, out_dir)
    report: Dict[str, Any] = {"rows_over_1pct": len(rows), "tables": paths, "diffs": {}}
    csv_dir = csv_dir or CSV_DIR_DEFAULT
    for name in ("degen_all_over1pct.csv", "degen_paper_over1pct.csv", "degen_glm_over1pct.csv"):
        p = os.path.join(csv_dir, name)
        if os.path.isfile(p):
            subset = {"degen_all_over1pct.csv": [r for r in rows if r.get("_paper_status") != "descoped"],
                      "degen_glm_over1pct.csv": [r for r in rows if r.get("_paper_status") == "descoped"],
                      "degen_paper_over1pct.csv": [r for r in rows if r.get("_paper_status") == "in_paper"]}[name]
            report["diffs"][name] = diff_against_csv(subset, p)
    with open(os.path.join(out_dir, "degeneracy_rebuild_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    return report


def cli_rebuild(args) -> int:
    from . import db as DB
    root = os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = rebuild(conn, args.out, getattr(args, "csv_dir", None))
    finally:
        conn.close()
    print(json.dumps(rep, indent=1, default=str))
    return 0
