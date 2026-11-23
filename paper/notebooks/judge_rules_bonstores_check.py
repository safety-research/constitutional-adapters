"""Prove `judge_rules`' StrongREJECT rule on the bon_stores lane from the PRODUCER's own per-draw scores.

Neither source `judge_rules_check.py` reads can prove this lane: the registry carries no item rows for the Claude (Haiku) judgment of
a bon_stores run, and the on-disk Haiku cache the re-judge's unit map pairs with is incomplete for these runs, so the item-row path
leaves most goals undetermined (`judge_rules.sr_guard_undetermined` refuses those readings). The producer graded every draw itself:

  * `scores_512.jsonl`: per (lane, cell, goal, sample_idx, trial) the store's draw at the 512-token cap, graded (`score_512`)
  * `analysis/scores_4096.jsonl`: the draws it climbed to 4,096 tokens, re-graded (`score_4096`, null when unscored; a null after a
    scored record of the same slot does not replace it)
  * `analysis/bon_cap4096_stats.json`: per cell the statistics a filed row's source_uri names (`›cells.<lane>/<cell>.asr_4096_cf_t<t>`)

A filed `asr_bon_cap4096` row is the counterfactual best-of-75 over those draws: per slot the 4,096 score where the slot climbed and
was scored, else its 512 score (a climbed slot left unscored keeps its 512 score, the producer's `n_4096_unscored_kept_512`). This
script builds the slots as judge_rules item rows (item_id = goal, trial_index = trial, sample_index = draw, value = the slot's score)
and applies the SAME functions the re-judge path applies, then compares with every filed row of the run's Claude judgment:

  * `judge_rules.sr_reading` at the row's threshold for k, n, value and the interval (both the producer row, computation_id '', and
    the export row that recomputes Wilson from k and n), and for the run's `asr_at_n<N>` rows (k, n, value, interval)
  * the row's extra grid `asr_at_n_4096_cf` (any-hit within the first N draws, at the row's threshold), as `sr_reading` re-derives it
  * the extra's `paired_vs_512` block (the counterfactual against the 512 store alone: delta, percentile bootstrap 10,000 from
    default_rng(0), exact binomial p, goals gained): a producer-side comparison, no judge_rules rule, checked so the slot map is the
    producer's
  * every run_contrast row of the run whose anchor is a bon_stores run too, through `judge_rules.sr_contrast` (delta, n_pairs, p and
    the interval within 0.75 pp)

    PY=.venv/bin/python                                     # from the repository root
    $PY paper/notebooks/judge_rules_bonstores_check.py [--bon-root DIR] [--out FILE.parquet]

`--bon-root` is the producer's bon_cap4096 directory: default env JUDGE_RULES_BON_ROOT, else `bon_cap4096` under the bon_stores
adapter's run-state tree (`sisteer.registry.adapters.lineage_common.RS_DEFAULT`, resolved by the repository's site configuration on
the host that graded the lane); with neither, the script stops and asks for `--bon-root`. The registry
(REGISTRY_ROOT, default `~/registry`) is opened read-only; no model is called. `judge_rules_check.py` runs the same check and files it
in its summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_rules as JR  # noqa: E402

DB = os.path.join(os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry"), "registry.sqlite")
LANE = "bon_stores"
CELL_KEY = re.compile(r"›cells\.(.+?)\.asr_4096_cf")
TOL = 1e-9
TOL_BOOT = 0.75


def default_root() -> str:
    """JUDGE_RULES_BON_ROOT, else the bon_stores adapter's run-state tree + bon_cap4096 (the repository's own site resolution); '' when
    neither resolves."""
    env = os.environ.get("JUDGE_RULES_BON_ROOT", "").strip()
    if env:
        return os.path.expanduser(env)
    src = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from sisteer.registry.adapters import lineage_common as LC
        return os.path.join(LC.expand(LC.RS_DEFAULT), "bon_cap4096")
    except Exception:                                          # no site config, or the package is not importable here
        return ""


def load_scores(root: str):
    """({(lane, cell): {(goal, sample, trial): score_512}}, {(lane, cell): {(goal, sample, trial): score_4096 or None}})."""
    s512: dict = collections.defaultdict(dict)
    with open(os.path.join(root, "scores_512.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s512[(r["lane"], r["cell_id"])][(str(r["goal_id"]), int(r["sample_idx"]), int(r["trial"]))] = r["score_512"]
    s4k: dict = collections.defaultdict(dict)
    with open(os.path.join(root, "analysis", "scores_4096.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cell = s4k[(r["lane"], r["cell_id"])]
            k = (str(r["goal_id"]), int(r["sample_idx"]), int(r["trial"]))
            if k in cell and cell[k] is not None and r["score_4096"] is None:
                continue
            cell[k] = r["score_4096"]
    return s512, s4k


def slot_items(cell512: dict, cell4k: dict):
    """The counterfactual slots of one store as judge_rules StrongREJECT unit rows, with (climbed, climbed but unscored) counts."""
    cf = dict(cell512)
    climbed = unscored = 0
    for k, v in cell4k.items():
        climbed += 1
        if v is None:
            unscored += 1
            continue
        cf[k] = v
    rows = [dict(item_id=g, metric_id=JR.SR_UNIT, trial_index=t, sample_index=s, turn_index=-1, value=float(v), label=None)
            for (g, s, t), v in cf.items()]
    return pd.DataFrame(rows), climbed, unscored


def store_items(cell512: dict) -> pd.DataFrame:
    """The 512 store alone as unit rows (the `paired_vs_512` baseline)."""
    return pd.DataFrame([dict(item_id=g, metric_id=JR.SR_UNIT, trial_index=t, sample_index=s, turn_index=-1, value=float(v), label=None)
                         for (g, s, t), v in cell512.items()])


def paired_vs_512(base: pd.Series, cf: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """The producer's `paired_vs_512` (analyze_bon_cap4096.py): per goal the counterfactual success minus the 512 success, percentile
    bootstrap over goals from default_rng(seed), exact two-sided binomial p on the goals gained vs lost."""
    from scipy.stats import binomtest
    a, b = base.to_numpy(float), cf.reindex(base.index).to_numpy(float)
    d = b - a
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1) * 100
    gained, lost = int(((b == 1) & (a == 0)).sum()), int(((a == 1) & (b == 0)).sum())
    m = gained + lost
    return dict(delta_pp=float(d.mean() * 100), lo_pp=float(np.percentile(boots, 2.5)), hi_pp=float(np.percentile(boots, 97.5)),
                goals_gained=gained, goals_lost=lost, mcnemar_exact_p=(float(binomtest(gained, m, 0.5).pvalue) if m else 1.0))


def _close(a, b, tol=TOL) -> bool:
    if (a is None or a != a) and (b is None or b != b):
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def p_agrees(mine, filed_p, tol=1e-6) -> bool:
    """A re-derived p against the filed one: equal within `tol`, or the test undefined on both readings (no discordant pair, every
    difference zero), which judge_rules reports as NaN and some producers file as 1.0 (declared_common.mcnemar_exact, this lane's
    analysis) while others file nothing (misalign.analyze, realign_stats)."""
    if _close(mine, filed_p, tol):
        return True
    return (mine is None or mine != mine) and _close(filed_p, 1.0, tol)


def filed(con) -> tuple:
    """(run_metric rows, run_contrast rows, run ids a re-judge judged) of the bon_stores runs' Claude judgments."""
    rm = pd.read_sql_query(
        "SELECT m.run_id, m.judgment_id, m.metric_id, m.threshold, m.computation_id, m.value, m.k, m.n, m.ci_lo, m.ci_hi, m.ci_method, "
        "m.source_uri, m.extra_json, j.judge_model FROM run_metric m JOIN run r USING(run_id) JOIN judgment g ON g.judgment_id = m.judgment_id "
        "JOIN judge j ON j.judge_id = g.judge_id WHERE r.lane_source_legacy = ? AND j.judge_model LIKE 'claude-%' AND g.status = 'complete' "
        "AND (m.metric_id = 'asr_bon_cap4096' OR m.metric_id LIKE 'asr_at_n%')", con, params=(LANE,))
    rc = pd.read_sql_query(
        "SELECT c.run_id, c.judgment_id, c.anchor_run_id, c.metric_id, c.threshold, c.delta, c.delta_lo, c.delta_hi, c.delta_p, c.n_pairs, "
        "c.delta_method FROM run_contrast c JOIN run r USING(run_id) JOIN judgment g ON g.judgment_id = c.judgment_id "
        "JOIN judge j ON j.judge_id = g.judge_id WHERE r.lane_source_legacy = ? AND j.judge_model LIKE 'claude-%'", con, params=(LANE,))
    rj = {r[0] for r in con.execute(
        "SELECT DISTINCT g.run_id FROM judgment g JOIN judge j USING(judge_id) JOIN run r USING(run_id) WHERE r.lane_source_legacy = ? "
        "AND j.judge_model NOT LIKE 'claude-%' AND g.status = 'complete'", (LANE,))}
    return rm, rc, rj


def verify_rows(rm: pd.DataFrame, rc: pd.DataFrame, s512: dict, s4k: dict, stats: dict | None = None, rejudged=frozenset()) -> tuple:
    """Pure core: the filed rows of the bon_stores runs (run_metric `rm`, run_contrast `rc`) against the producer's scores. Returns
    (per filed reading row: k / n / value / interval / grid ok, per contrast row: delta / n_pairs / p / interval ok)."""
    keys = {}
    for r in rm[(rm.metric_id == "asr_bon_cap4096") & rm.computation_id.fillna("").eq("")].itertuples():
        m = CELL_KEY.search(str(r.source_uri or ""))
        if m:
            keys[r.run_id] = m.group(1)
    items: dict = {}
    out = []
    for r in rm.itertuples():
        key = keys.get(r.run_id)
        rec = dict(run_id=r.run_id, key=key, metric_id=r.metric_id, threshold=r.threshold, computation_id=r.computation_id or "",
                   rejudged=r.run_id in rejudged, filed_k=r.k, filed_n=r.n, filed_value=r.value, why="")
        if key is None:
            rec["why"] = "no producer cell key on the run's asr_bon_cap4096 row"
            out.append(rec)
            continue
        lane, cell = key.split("/", 1)
        if (lane, cell) not in s512:
            rec["why"] = f"no 512-store scores for {key}"
            out.append(rec)
            continue
        if r.run_id not in items:
            it, climbed, unscored = slot_items(s512[(lane, cell)], s4k.get((lane, cell), {}))
            items[r.run_id] = (it, climbed, unscored)
        it, climbed, unscored = items[r.run_id]
        st = ((stats or {}).get("cells") or {}).get(key) or {}
        rec.update(n_slots=len(it), n_goals=it.item_id.nunique(), n_climbed=climbed, n_climbed_unscored=unscored,
                   stats_n_climbed=st.get("n_climbed"))
        ctx = JR.row_context(dict(eval_id="strongreject_bon", metric_id=r.metric_id, threshold=r.threshold, ci_method=r.ci_method,
                                  lane_source_legacy=LANE, bon_n=75, metric_extra_json=r.extra_json))
        try:
            o = JR.sr_reading(ctx, it)
        except JR.NotDerivable as e:
            rec["why"] = f"not derivable: {e}"
            out.append(rec)
            continue
        rec.update(k=o["k_effective"], n=o["n"], value=o["value"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"],
                   k_ok=_close(o["k_effective"], r.k) or r.k != r.k, n_ok=_close(o["n"], r.n) or r.n != r.n, value_ok=_close(o["value"], r.value),
                   ci_ok=(_close(o["ci_lo"], r.ci_lo, 1e-6) and _close(o["ci_hi"], r.ci_hi, 1e-6)) or r.ci_lo != r.ci_lo)
        ex = json.loads(r.extra_json) if isinstance(r.extra_json, str) and r.extra_json.strip().startswith("{") else {}
        grid = ex.get(JR.BON_GRID) if isinstance(ex, dict) else None
        if isinstance(grid, dict) and o.get("extra"):
            mine = json.loads(o["extra"]).get(JR.BON_GRID) or {}
            rec["grid_ok"] = all(_close(mine.get(k), v, 1e-12) for k, v in grid.items()) and set(mine) == set(grid)
        pv = ex.get("paired_vs_512") if isinstance(ex, dict) else None
        if isinstance(pv, dict):
            thr = float(r.threshold)
            base = (store_items(s512[(lane, cell)]).groupby("item_id")["value"].max() > thr).astype(float).sort_index()
            p = paired_vs_512(base, (it.groupby("item_id")["value"].max() > thr).astype(float))
            rec["paired_vs_512_ok"] = (all(_close(p[x], pv.get(x)) for x in ("delta_pp", "lo_pp", "hi_pp", "mcnemar_exact_p"))
                                       and p["goals_gained"] == pv.get("goals_gained"))
        out.append(rec)
    crows = []
    for c in rc.itertuples():
        rec = dict(run_id=c.run_id, anchor_run_id=c.anchor_run_id, metric_id=c.metric_id, threshold=c.threshold,
                   delta_method=c.delta_method, filed_delta=c.delta, filed_n_pairs=c.n_pairs, filed_delta_p=c.delta_p, why="")
        if c.run_id not in items or c.anchor_run_id not in keys:
            rec["why"] = "the anchor is not a bon_stores store with producer scores" if c.run_id in items else "no producer scores for the run"
            crows.append(rec)
            continue
        if c.anchor_run_id not in items:
            lane, cell = keys[c.anchor_run_id].split("/", 1)
            if (lane, cell) not in s512:
                rec["why"] = f"no 512-store scores for the anchor {keys[c.anchor_run_id]}"
                crows.append(rec)
                continue
            items[c.anchor_run_id] = slot_items(s512[(lane, cell)], s4k.get((lane, cell), {}))
        ctx = JR.row_context(dict(eval_id="strongreject_bon", metric_id=c.metric_id, threshold=c.threshold, delta_method=c.delta_method,
                                  lane_source_legacy=LANE, run_id=c.run_id, anchor_run_id=c.anchor_run_id))
        try:
            o = JR.sr_contrast(ctx, items[c.run_id][0], items[c.anchor_run_id][0])
        except JR.NotDerivable as e:
            rec["why"] = f"not derivable: {e}"
            crows.append(rec)
            continue
        rec.update(delta=o["delta"], n_pairs=o["n_pairs"], delta_p=o.get("delta_p"), delta_lo=o.get("delta_lo"), delta_hi=o.get("delta_hi"),
                   delta_ok=_close(o["delta"], c.delta, 1e-6), n_pairs_ok=_close(o["n_pairs"], c.n_pairs) or c.n_pairs != c.n_pairs,
                   p_ok=p_agrees(o.get("delta_p"), c.delta_p) or c.delta_p != c.delta_p,
                   ci_ok=(_close(o.get("delta_lo"), c.delta_lo, TOL_BOOT) and _close(o.get("delta_hi"), c.delta_hi, TOL_BOOT)) or c.delta_lo != c.delta_lo)
        crows.append(rec)
    R, C = pd.DataFrame(out), pd.DataFrame(crows)
    if len(R):
        ok = [c for c in ("k_ok", "n_ok", "value_ok", "ci_ok") if c in R.columns]
        R["match_all"] = R.why.eq("") & R[ok].fillna(False).astype(bool).all(axis=1) & R.get("grid_ok", pd.Series(True, index=R.index)).fillna(True).astype(bool)
    if len(C):
        ok = [c for c in ("delta_ok", "n_pairs_ok", "p_ok", "ci_ok") if c in C.columns]
        C["match_all"] = C.why.eq("") & C[ok].fillna(False).astype(bool).all(axis=1)
    return R, C


def verify(con, root: str) -> tuple:
    """Load the producer files under `root` and the filed rows from the registry connection `con`; `verify_rows` does the rest."""
    s512, s4k = load_scores(root)
    sp = os.path.join(root, "analysis", "bon_cap4096_stats.json")
    stats = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
    rm, rc, rj = filed(con)
    return verify_rows(rm, rc, s512, s4k, stats, rj)


def summary_lines(R: pd.DataFrame, C: pd.DataFrame) -> list:
    """Markdown lines: per (metric family, threshold, computation) the filed rows reproduced, and the contrasts."""
    if not len(R):
        return ["No bon_stores filed rows were found."]
    R = R.assign(metric=np.where(R.metric_id.eq("asr_bon_cap4096"), "asr_bon_cap4096", "asr_at_n<N>"))
    agg = {"rows": ("run_id", "size"), "runs": ("run_id", "nunique"), "not_derivable": ("why", lambda s: int(s.ne("").sum()))}
    for c in ("k_ok", "n_ok", "value_ok", "ci_ok", "match_all"):
        if c in R.columns:
            agg[c] = (c, lambda s: int(s.fillna(False).astype(bool).sum()))
    for c in ("grid_ok", "paired_vs_512_ok"):            # "ok of checked": only the producer rows carry the grid and the 512 comparison
        if c in R.columns:
            agg[c] = (c, lambda s: f"{int(s.fillna(False).astype(bool).sum())} of {int(s.notna().sum())}")
    g = R.groupby(["metric", "threshold", "computation_id", "rejudged"], dropna=False).agg(**agg).reset_index()
    lines = [_md(g), ""]
    head = R[R.metric.eq("asr_bon_cap4096")]
    lines.append(f"Headline rows: k reproduced on {int(head.get('k_ok', pd.Series(dtype=bool)).fillna(False).sum())} of {len(head)} "
                 f"({head.run_id.nunique()} runs; the runs a re-judge judged: {int(head[head.rejudged].get('k_ok', pd.Series(dtype=bool)).fillna(False).sum())} "
                 f"of {int(head.rejudged.sum())} rows, {head[head.rejudged].run_id.nunique()} runs); every field (k, n, value, interval, grid): "
                 f"{int(head.match_all.sum())} of {len(head)}.")
    if "n_climbed" in R.columns:
        u = R.drop_duplicates("run_id")
        lines.append(f"Climbed slots: {int(u.n_climbed.fillna(0).sum())} over {len(u)} runs, {int(u.n_climbed_unscored.fillna(0).sum())} left unscored "
                     f"(keep their 512 score); the climb counts equal the statistics file's on "
                     f"{int((u.n_climbed == u.stats_n_climbed).sum())} of {len(u)} runs.")
    if len(C):
        agg = {"rows": ("run_id", "size"), "not_derivable": ("why", lambda s: int(s.ne("").sum()))}
        for c in ("delta_ok", "n_pairs_ok", "p_ok", "ci_ok", "match_all"):
            if c in C.columns:
                agg[c] = (c, lambda s: int(s.fillna(False).astype(bool).sum()))
        gc = C.assign(method=C.delta_method.fillna("").str.slice(0, 60)).groupby(["metric_id", "threshold", "method"], dropna=False).agg(**agg).reset_index()
        lines += ["", "Contrasts through `judge_rules.sr_contrast` on the producer's slots (the interval within 0.75 pp):", "", _md(gc)]
    bad = R[~R.match_all]
    if len(bad):
        lines += ["", "Rows not reproduced:", "", _md(bad[[c for c in ("run_id", "metric_id", "threshold", "computation_id", "filed_k", "k", "filed_n", "n",
                                                              "why") if c in bad.columns]])]
    return lines


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    fmt = lambda v: "" if v is None or (isinstance(v, float) and v != v) else (f"{v:.6g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bon-root", default="", help="the producer's bon_cap4096 directory (default: JUDGE_RULES_BON_ROOT, else the adapter's)")
    ap.add_argument("--out", default="", help="write the per-row table here (parquet); its contrasts beside it (<stem>_contrasts.parquet)")
    a = ap.parse_args()
    root = os.path.expanduser(a.bon_root) if a.bon_root else default_root()
    if not root or not os.path.isdir(root):
        sys.exit(f"no bon_cap4096 directory ({root or 'unresolved'}): pass --bon-root or set JUDGE_RULES_BON_ROOT")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    R, C = verify(con, root)
    if a.out:
        R.to_parquet(a.out)
        C.to_parquet(os.path.splitext(a.out)[0] + "_contrasts.parquet")
    print("\n".join(summary_lines(R, C)))
    sys.exit(0 if len(R) and bool(R[R.metric_id.eq("asr_bon_cap4096")].match_all.all()) else 1)
