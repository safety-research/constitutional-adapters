"""`registry export r2-compat`: the deprecated round-2 export's columns from the registry.

Renders the export's 82 columns (the parquet the peers' notebooks read) from the rows the `r2_export` adapter
wrote — `run` + `run_metric` + `run_contrast` + `run_degeneration` for measured rows, `disposition` rows for
the unmeasured ones — regenerates `row_id` exactly as the builder does (sort, then `range(len)`), and writes
the parquet, the csv, `_schema.json` (the builder's SCHEMA dict, vendored verbatim under `data/r2_schema.json`) and
`_sources.json` (the artifact rows the adapter registered, with their shas and the registry head).

`--compare <parquet>` joins the rendered frame with an existing export on the builder's own join key
`(view, eval_id, cell_id, metric, threshold, decode_cap_tokens, pass, bon_n)` and reports per-column
differences (numeric to 1e-9, strings exact, NaN == NaN), rows only on either side, and never asserts.

Deprecated surface: the export exists so the notebooks keep working through round r2; the status page lists it
under the retirement notice.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .adapters import lineage_common as LC
from .adapters.r2_export import COLUMN_MAP, SAMPLED_SUFFIX

R2_COLUMNS: List[str] = list(COLUMN_MAP)          # the parquet's column order
SORT_KEYS = ["dose", "model", "view", "cell_id", "component", "eval_id", "metric", "threshold", "bon_n"]
SORT_ASCENDING = [False, True, True, True, True, True, True, True, True]
JOIN_KEY = ["view", "eval_id", "cell_id", "metric", "threshold", "decode_cap_tokens", "pass", "bon_n"]
#: The builder's key is not unique over its own export (68 keys fork on the provenance json path, e.g. the GLM recovered arms
#: `arms.t0` vs `arms.sampled`), so rows are MATCHED on it plus `provenance`; duplicates on both keys are reported.
MATCH_KEY = JOIN_KEY + ["provenance"]
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "r2_schema.json")
DEPRECATION = ("DEPRECATED (registry decision 21): this file is rendered FROM the results registry for round r2's notebooks; "
               "read the registry (v_paper_long, run_metric) instead. Retirement: the round after r2.")


def _j(text) -> Dict[str, Any]:
    if text is None:
        return {}
    if isinstance(text, dict):
        return text
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _nan(v):
    return float("nan") if v is None else v


def _model_key_for(setting: str) -> str:
    return setting[:-len("_halfdose")] if setting.endswith("_halfdose") else setting


def _dose_for(setting: str) -> float:
    return 0.5 if setting.endswith("_halfdose") else 1.0


def _effect_kind_prose(kind: Optional[str], extra: Mapping[str, Any]) -> Optional[str]:
    return extra.get("effect_kind") or kind


def render_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """One dict per export row, in the parquet's columns (no row_id yet)."""
    conn.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    runs = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM run").fetchall()}
    contrasts: Dict[Tuple[str, str, str, float, str], List[Dict[str, Any]]] = {}
    for c in conn.execute("SELECT * FROM run_contrast").fetchall():
        c = dict(c)
        contrasts.setdefault((c["run_id"], c["judgment_id"], c["metric_id"], float(c["threshold"]), c["computation_id"]), []).append(c)
    n_metric_rows = 0
    for m in conn.execute("SELECT * FROM run_metric").fetchall():
        m = dict(m)
        extra = _j(m.get("extra_json"))
        if "row_id" not in extra or "status_legacy" not in extra:
            continue   # not an r2 export row (another adapter's metric on a shared run)
        if "refused_rate_twin" in extra:
            continue   # the registry's refused-rate TWIN of an export row (r2_export 9): the source row itself is rendered; the twin is not a second export row
        n_metric_rows += 1
        run = runs.get(m["run_id"]) or {}
        # the row's own view spelling: one run (one store) can be filed under two views — the over-refusal controls anchor both
        # the base→instruct and the instruct lanes — so the run's view_legacy is the fallback, not the value
        setting = extra.get("view_legacy_row") or run.get("view_legacy") or extra.get("view")
        row: Dict[str, Any] = {c: float("nan") for c in R2_COLUMNS}
        # parked columns first (by name), then the typed ones on top. Row-level spellings the run ALSO carries (pass, judge, decode,
        # grades_root, petri_run_key, served_model, lane_kind, …) come from the row's park only: a run is shared by rows of two
        # views / passes and by another adapter's record of the same run, so the run-level value is not the row's.
        for col in R2_COLUMNS:
            if col in extra:
                row[col] = LC.json_unsafe(extra[col])
        row.update({
            "view": setting, "model": _model_key_for(setting or ""), "dose": _dose_for(setting or ""), "cell_id": run.get("campaign_cell_id_legacy"),
            "eval_id": extra.get("eval_id_legacy"), "metric": m.get("producer_metric_spelling") or m["metric_id"],
            "threshold": float("nan") if m["threshold"] == -1 else m["threshold"],
            "value": _nan(m.get("value")), "k": _nan(m.get("k")), "n": _nan(m.get("n")), "ci_lo": _nan(m.get("ci_lo")), "ci_hi": _nan(m.get("ci_hi")),
            "ci_method": m.get("ci_method"), "at_cap_share": _nan(m.get("frac_at_token_cap")), "loop_share": _nan(m.get("frac_degenerate_rep_ratio_loop")),
            "loop_share_first512": _nan(m.get("frac_loop_first512")), "abstentions": _nan(m.get("n_abstentions")), "n_error_outcomes": _nan(m.get("n_error_outcomes")),
            "status": extra.get("status_legacy"),
            "provenance": (f"{m['source_uri']} ›{extra['provenance_json_path']}" if m.get("source_uri") and extra.get("provenance_json_path")
                           else (m.get("source_uri") or float("nan"))),
        })
        if "decode_cap_tokens" in extra:
            row["decode_cap_tokens"] = extra["decode_cap_tokens"]
        # defended_degenerate_k / defended_n are PARKED on every row that carries them (verified live: per cell, every row of the
        # cell); a fill from the run's degeneration row would also pick up the union adapter's reading and invent a value where
        # the export has none — so no fill.
        # contrasts: the none-anchor one (delta*), the full-dose one (delta_vs_full_dose*)
        for c in contrasts.get((m["run_id"], m["judgment_id"], m["metric_id"], float(m["threshold"]), m["computation_id"]), []):
            cx = _j(c.get("extra_json"))
            if c.get("anchor_kind") == "full_dose":
                row["delta_vs_full_dose"], row["delta_vs_full_dose_lo"], row["delta_vs_full_dose_hi"] = _nan(c.get("delta")), _nan(c.get("delta_lo")), _nan(c.get("delta_hi"))
            else:
                row.update(delta=_nan(c.get("delta")), delta_lo=_nan(c.get("delta_lo")), delta_hi=_nan(c.get("delta_hi")), delta_p=_nan(c.get("delta_p")),
                           delta_method=_nan(cx["delta_method_legacy"]) if "delta_method_legacy" in cx else c.get("delta_method"),
                           effect_size=_nan(c.get("effect_size")),
                           effect_kind=_nan(extra.get("effect_kind") or cx.get("effect_kind_legacy") or c.get("effect_kind")))
                for k in ("effect_lo", "effect_hi", "pairing_allowance", "anchor_engine", "anchor_root"):
                    if cx.get(k) is not None:
                        row[k] = cx[k]
                if cx.get("anchor_view_legacy") is not None and extra.get("anchor_view") is not None:
                    row["anchor_view"] = extra["anchor_view"]
        blk = extra.get("delta_block")
        if isinstance(blk, dict):
            for k in ("delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind", "effect_lo", "effect_hi"):
                if blk.get(k) is not None:
                    row[k] = blk[k]
        # a half-dose row whose `full_dose` comparison parked (its 1x anchor not a current run, or not folded yet; r2_export 11): the
        # producer's `delta_vs_full_dose*` ride the park, so the round trip renders them whether the contrast was filed or parked
        fd = extra.get("delta_vs_full_dose_block")
        if isinstance(fd, dict):
            for k, col in (("delta", "delta_vs_full_dose"), ("delta_lo", "delta_vs_full_dose_lo"), ("delta_hi", "delta_vs_full_dose_hi")):
                if fd.get(k) is not None:
                    row[col] = fd[k]
        out.append(row)
    # the export's DEGENERATION statistics (metric rows per cell on the parquet) live in run_degeneration; each row's parquet columns
    # are parked whole under the reading's `extra.r2_rows` (one entry per spelling the reading was filed under), so the deprecated
    # export renders them back as the rows they were
    for d in conn.execute("SELECT * FROM v_run_degeneration_current").fetchall():
        d = dict(d)
        extra = _j(d.get("extra_json"))
        run = runs.get(d["run_id"]) or {}
        for park in extra.get("r2_rows") or []:
            if not isinstance(park, dict) or "row_id" not in park:
                continue
            setting = park.get("view_legacy_row") or run.get("view_legacy") or park.get("view")
            row = {c: float("nan") for c in R2_COLUMNS}
            for col in R2_COLUMNS:
                if col in park:
                    row[col] = LC.json_unsafe(park[col])
            row.update({"view": setting, "model": _model_key_for(setting or ""), "dose": _dose_for(setting or ""), "cell_id": run.get("campaign_cell_id_legacy"),
                        "eval_id": park.get("eval_id_legacy"), "metric": park.get("producer_metric_spelling"), "status": park.get("status_legacy"),
                        "provenance": (f"{park['source_uri']} ›{park['provenance_json_path']}" if park.get("source_uri") and park.get("provenance_json_path")
                                       else (park.get("source_uri") or float("nan")))})
            out.append(row)
    for d in conn.execute("SELECT * FROM disposition").fetchall():
        d = dict(d)
        extra = _j(d.get("extra_json"))
        if "row_id" not in extra:
            continue
        row = {c: float("nan") for c in R2_COLUMNS}
        for col in R2_COLUMNS:
            if col in extra:
                row[col] = LC.json_unsafe(extra[col])
        row["status"] = extra.get("status_legacy", row.get("status"))
        out.append(row)
    return out


def render_frame(conn: sqlite3.Connection):
    import pandas as pd
    rows = render_rows(conn)
    cols = [c for c in R2_COLUMNS if c != "row_id"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        df.insert(0, "row_id", pd.Series(dtype="int64"))
        return df
    df = df.sort_values(SORT_KEYS, ascending=SORT_ASCENDING, kind="stable", na_position="last").reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))
    return df


def write_export(conn: sqlite3.Connection, out_parquet: str, now_utc: str) -> Dict[str, Any]:
    df = render_frame(conn)
    base = out_parquet[:-len(".parquet")] if out_parquet.endswith(".parquet") else out_parquet
    os.makedirs(os.path.dirname(os.path.abspath(out_parquet)) or ".", exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(base + ".csv", index=False)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    schema["_deprecation"] = DEPRECATION
    with open(base + "_schema.json", "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=1)
    head = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    arts = {}
    for a in conn.execute("SELECT artifact_id, kind, uri_primary, bytes, verified_utc, extra_json FROM artifact WHERE record_uri LIKE '%adapters/r2_export/%'").fetchall():
        arts[a[2]] = {"sha256": a[0], "kind": a[1], "bytes": a[3], "verified_utc": a[4]}
    sources = {"built_utc": now_utc, "n_rows": int(len(df)), "n_measured_rows": int(df["value"].notna().sum()) if len(df) else 0,
               "registry_head_ingest_seq": head, "deprecation": DEPRECATION, "sources": arts}
    with open(base + "_sources.json", "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=1, default=str)
    return {"rows": int(len(df)), "measured": sources["n_measured_rows"], "columns": int(len(df.columns)), "artifacts": len(arts),
            "files": [out_parquet, base + ".csv", base + "_schema.json", base + "_sources.json"]}


# --------------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------------

def _key_frame(df, cols=None):
    import pandas as pd
    cols = list(cols or JOIN_KEY)
    k = df.copy()
    for c in cols:
        if c not in k.columns:
            k[c] = float("nan")
    key = k[cols].astype(object).where(k[cols].notna(), "<NA>")
    key["threshold"] = key["threshold"].map(lambda v: v if v == "<NA>" else f"{float(v):.6g}")
    key["decode_cap_tokens"] = key["decode_cap_tokens"].map(lambda v: v if v == "<NA>" else str(int(float(v))))
    key["bon_n"] = key["bon_n"].map(lambda v: v if v == "<NA>" else str(int(float(v))))
    return key.astype(str).agg("|".join, axis=1)


def compare(rendered, reference, tol: float = 1e-9) -> Dict[str, Any]:
    """Per-key, per-column differences between the rendered frame and a reference parquet frame."""
    import numpy as np
    import pandas as pd
    a = rendered.copy()
    b = reference.copy()
    dup_a = int(_key_frame(a, JOIN_KEY).duplicated().sum())
    dup_b = int(_key_frame(b, JOIN_KEY).duplicated().sum())
    a["_key"] = _key_frame(a, MATCH_KEY)
    b["_key"] = _key_frame(b, MATCH_KEY)
    dup_a_m = int(a["_key"].duplicated().sum())
    dup_b_m = int(b["_key"].duplicated().sum())
    a = a.drop_duplicates("_key").set_index("_key")
    b = b.drop_duplicates("_key").set_index("_key")
    common = a.index.intersection(b.index)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    cols = [c for c in R2_COLUMNS if c != "row_id" and c in b.columns]
    per_col: Dict[str, Dict[str, Any]] = {}
    for c in cols:
        if c not in a.columns:
            per_col[c] = {"missing_in_rendered": True}
            continue
        x = a.loc[common, c]
        y = b.loc[common, c]
        if pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            xn = pd.to_numeric(x, errors="coerce").astype(float)
            yn = pd.to_numeric(y, errors="coerce").astype(float)
            both_nan = xn.isna() & yn.isna()
            diff = (xn - yn).abs()
            bad = ~both_nan & ~((diff <= tol) | (xn == yn))   # inf == inf: their difference is nan, the values are equal
            finite = diff[~both_nan & (xn != yn)].dropna()
            per_col[c] = {"n_compared": int(len(common)), "n_differ": int(bad.sum()),
                          "max_abs_diff": float(finite.max()) if len(finite) else 0.0,
                          "examples": [str(k) for k in common[bad.to_numpy()][:3]]}
        else:
            xs = x.astype(object).where(x.notna(), None)
            ys = y.astype(object).where(y.notna(), None)
            bad = ~((xs == ys) | (xs.isna() & ys.isna()))
            per_col[c] = {"n_compared": int(len(common)), "n_differ": int(bad.sum()), "examples": [str(k) for k in common[bad.to_numpy()][:3]]}
    return {"rendered_rows": int(len(rendered)), "reference_rows": int(len(reference)), "common_keys": int(len(common)),
            "only_rendered": int(len(only_a)), "only_reference": int(len(only_b)),
            "duplicate_join_keys_rendered": dup_a, "duplicate_join_keys_reference": dup_b,
            "duplicate_match_keys_rendered": dup_a_m, "duplicate_match_keys_reference": dup_b_m, "match_key": MATCH_KEY,
            "only_reference_examples": [str(k) for k in only_b[:8]], "only_rendered_examples": [str(k) for k in only_a[:8]],
            "columns_differing": {c: v for c, v in per_col.items() if v.get("n_differ", 0) or v.get("missing_in_rendered")},
            "columns_equal": [c for c, v in per_col.items() if not v.get("n_differ", 0) and not v.get("missing_in_rendered")]}


def cli_export(args) -> int:
    """`registry export r2-compat OUT.parquet [--compare REF.parquet]`."""
    from . import db as DB
    from . import records as R
    root = os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = write_export(conn, args.out, R.utc_now())
        if getattr(args, "compare", None):
            import pandas as pd
            rep["compare"] = compare(render_frame(conn), pd.read_parquet(args.compare))
    finally:
        conn.close()
    print(json.dumps(rep, indent=1, default=str))
    print(DEPRECATION, file=__import__("sys").stderr)
    return 0
