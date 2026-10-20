"""Stage-2 fixed-dose panel: every arm at strength 1.0.

The headline presentation: instead of the frontier's per-cell
best-in-budget scale (which needs F4 to de-curse), report every
(SI, arm) cell at the SAME pre-registered dose — scale 1.0, the
trained object exactly as learned. Nothing is selected, so there is
no winner's-curse on scale by construction; the only shared caveat
left is the common 100-prompt eval slice.

Per (instruction, arm) cell:
1. on-target effect = strict(steered@1.0) − strict(prompted), paired
   per-prompt bootstrap CI (2000 resamples, seed 0 — the s2_analysis
   convention)
2. off-target deltas vs UNSTEERED at scale 1.0: MMLU, GSM8K, GPQA-d,
   MMLU-Pro — reusing every measurement that already exists (capability
   panels, F1 MMLU fill-ins in frontier.json, GSM8K fill-ins) and
   measuring the rest with the stored object (write-through cache
   dose1_fillins.json, so a walltime kill resumes).

Writes <out-dir>/dose1.json:
  {si: {arm: {steered, prompted, effect, ci_lo, ci_hi, n_prompts,
              mmlu/gsm8k/gpqa_diamond/mmlu_pro (+ *_delta vs unsteered),
              sources: {metric: panel|frontier|fillin-cache|measured}}}}
plus a "_bases" entry with the unsteered reference values.

    uv run python scripts/stage2_dose1.py \
        --out-dir <.../stage2_llama8b/ifeval> \
        --config configs/stage2_llama8b.yaml [--dry]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage2_frontier import (  # noqa: E402
    _ext_metric,
    _load_cell_object,
    _write_json_atomic,
)

DOSE = 1.0
METRICS = ("mmlu", "gsm8k", "gpqa_diamond", "mmlu_pro")
N_BOOT = 2000


def _prompt_strict(res: dict, arm: str, scale: float):
    """Per-prompt strict outcomes for an arm entry (s2_analysis pattern):
    (values, prompts), trigger rows only."""
    for i, a in enumerate(res["arms"]):
        if a["arm"] == arm and abs(a.get("scale", 0.0) - scale) < 1e-9:
            rows = [r for r in res["records"][i] if not r.get("nontrigger")]
            return [r["strict"] for r in rows], [r["prompt"] for r in rows]
    raise KeyError(f"no {arm}@{scale:g} arm entry")


def paired_boot(diffs: np.ndarray, seed: int = 0, n_boot: int = N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(means, 0.025)), \
        float(np.quantile(means, 0.975))


def existing_value(res: dict, frontier_cell: dict | None, fills: dict,
                   si: str, arm: str, metric: str):
    """Reuse any already-paid measurement of <metric> at scale 1.0:
    the cell's capability panel, the F1 MMLU fill-ins recorded in
    frontier.json candidates, or the GSM8K fill-in cache."""
    v = _ext_metric(res, DOSE, metric)
    if v is not None:
        return v, "panel"
    if metric == "mmlu":
        # the stage-1-shape capability block also carries panel MMLU
        v = (res.get("capability", {}).get(f"scale_{DOSE}") or {}).get("mmlu")
        if v is not None:
            return v, "panel"
    if metric == "mmlu" and frontier_cell:
        for c in frontier_cell.get("candidates", []):
            if abs(c["scale"] - DOSE) < 1e-9 and c.get("mmlu") is not None:
                return c["mmlu"], "frontier"
    if metric == "gsm8k":
        v = fills.get(f"{si}|{arm}|{DOSE:g}")
        if v is not None:
            return v, "fillin-cache"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry", action="store_true",
                    help="reuse-only: no model, unmeasured metrics stay null")
    args = ap.parse_args()
    out = Path(args.out_dir)

    frontier = {}
    fpath = out / "frontier.json"
    if fpath.exists():
        frontier = json.loads(fpath.read_text())
    gpath = out / "frontier_gsm8k_fillins.json"
    gsm8k_fills = json.loads(gpath.read_text()) if gpath.exists() else {}
    cache_path = out / "dose1_fillins.json"
    cache: dict = (json.loads(cache_path.read_text())
                   if cache_path.exists() else {})

    model = tok = None
    rows_by_metric: dict = {}
    if not args.dry:
        import yaml

        from sisteer.eval import capability_ext as ce
        from stage2_run import _load_model

        cfg = yaml.safe_load(Path(args.config).read_text()) if args.config \
            else {}
        if not cfg.get("model"):
            raise SystemExit("--config with a pinned model is required "
                             "when not --dry (D22#10)")
        model, tok = _load_model(cfg)
        rows_by_metric = {
            "mmlu": ce._mmlu_rows(),
            "gsm8k": ce._gsm8k_rows(n=200),
            "gpqa_diamond": ce.gpqa_diamond_rows(),
            "mmlu_pro": ce.mmlu_pro_rows(n=300),
        }

    def measure(metric, obj):
        from sisteer.eval import capability_ext as ce

        rows = rows_by_metric[metric]
        if metric == "mmlu":
            return ce.mmlu_under_obj(model, tok, obj, scale=DOSE, rows=rows)
        if metric == "gsm8k":
            return ce.gsm8k_accuracy(model, tok, obj, scale=DOSE,
                                     rows=rows)["acc"]
        if metric == "gpqa_diamond":
            return ce.gpqa_diamond_accuracy(model, tok, obj, scale=DOSE,
                                            rows=rows)
        return ce.mmlu_pro_accuracy(model, tok, obj, scale=DOSE, rows=rows)

    cells = sorted(p.parent for p in out.glob("*/*/results.json"))
    dose1: dict = {}
    bases: dict = {}
    for cell in cells:
        res = json.loads((cell / "results.json").read_text())
        si = res.get("instruction", cell.parent.name)
        arm = res.get("method", cell.name)
        s_vals, s_prompts = _prompt_strict(res, "steered", DOSE)
        p_vals, p_prompts = _prompt_strict(res, "prompted", 0.0)
        assert s_prompts == p_prompts, (si, arm)
        eff, lo, hi = paired_boot(np.array(s_vals, float)
                                  - np.array(p_vals, float))
        entry = {
            "steered": float(np.mean(s_vals)),
            "prompted": float(np.mean(p_vals)),
            "effect": round(eff, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "n_prompts": len(s_vals),
            "sources": {},
        }
        fcell = (frontier.get(si) or {}).get(arm)
        obj = None
        for metric in METRICS:
            # unsteered base: object-free, model-global — first panel wins
            if metric not in bases:
                b = _ext_metric(res, 0.0, metric)
                if b is None and metric == "mmlu":
                    b = (res.get("capability", {})
                         .get("unsteered", {}).get("mmlu"))
                if b is not None:
                    bases[metric] = b
            v, src = existing_value(res, fcell, gsm8k_fills, si, arm, metric)
            key = f"{si}|{arm}|{metric}"
            if v is None:
                v, src = cache.get(key), "fillin-cache"
                if v is None:
                    src = None
            if v is None and not args.dry:
                if obj is None:
                    obj = _load_cell_object(cell, model, tok)
                v, src = measure(metric, obj), "measured"
                cache[key] = v
                _write_json_atomic(cache_path, cache)
            entry[metric] = v
            entry["sources"][metric] = src
            base = bases.get(metric)
            entry[f"{metric}_delta"] = (round(v - base, 4)
                                        if v is not None and base is not None
                                        else None)
        dose1.setdefault(si, {})[arm] = entry
        print(f"[dose1] {si}/{arm}: effect {eff:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]  "
              + " ".join(f"{m}={entry[m] if entry[m] is None else round(entry[m], 3)}"
                         for m in METRICS))

    dose1["_bases"] = bases
    _write_json_atomic(out / "dose1.json", dose1)
    print(f"wrote {out / 'dose1.json'} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
