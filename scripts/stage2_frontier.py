"""Stage-2 post-processing: the within-budget compliance frontier (F1).

Mirror of scripts/stage1_frontier.py, generalized to the stage-2 layout:
cells live at <out-dir>/<instruction>/<method>/results.json where
<method> is any of the 12 arm objects (or rung2/rung3 under
--ladder-dir), results.json carries "arms"/"records"/"capability" as
written by scripts/stage2_run.py, and the intervention artifact is
re-hydrated through methods.load_object (content-sniffed, moved to the
model device) so the MMLU fill-in runs the SAME object the cell ran.

Per (instruction, method):
1. candidate scales = all steered scales within 0.9x of the cell's best
   strict compliance (and > 0);
2. MMLU at each candidate not already measured (reusing the stored
   object; skipped under --dry);
3. distinct-3gram repetition of each candidate's recorded responses
   (free — from results.json);
4. budget (superseding the stage-1-only rule): MMLU >= unsteered
   - 0.02 AND distinct3 >= 0.35 AND GSM8K >= unsteered - 0.05. The GSM8K
   bound is 5pt, not MMLU's 2pt, because GSM8K is generative at n=200
   (SE ~3.5pt — a 2pt bound would exclude cells by noise); candidate
   GSM8K values are all recorded in frontier.json so a 2pt sensitivity
   map derives without re-running. --legacy-budget reproduces the
   MMLU-only stage-1-comparable map (never into frontier.json — it
   writes frontier.legacy.json).
5. frontier = best compliance among in-budget scales.

GSM8K fill-ins are GENERATIVE (200 completions x 512 tokens per
candidate scale, vs MMLU's logprob pass), so they are cached
write-through in <out-dir>/frontier_gsm8k_fillins.json keyed
"<instr>|<method>|<scale>" — a walltime kill resumes without re-paying
completed measurements. The unsteered GSM8K base is object-free and so
model-global: cached once under "unsteered" when a cell's panel lacks it.

Writes <out-dir>/frontier.json (the shape judge_batch.py's pick_arms and
the qwen-spot arm selection read) and prints the instruction x method
matrix. Two rules:

- --dry NEVER touches frontier.json: it writes frontier.preview.json
  only, so a parse-only rerun can't clobber a real frontier that
  downstream cells already consumed.
- a candidate whose MMLU was never measured is "unknown"
  (in_budget=null), never treated as in-budget — the frontier picks only
  among candidates whose budget check actually PASSED.

Model contract: pass --config <experiment yaml> and the model, pinned
model_revision, AND dtype are consumed from it through the SAME loader
stage2_run uses — so the MMLU fill-in runs the exact snapshot the matrix
trained on. --model/--revision/--dtype override individually.

    uv run python scripts/stage2_frontier.py \
        --out-dir <.../stage2_llama8b/ifeval> \
        --config configs/stage2_llama8b.yaml [--ladder-dir <...>] [--dry]

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sisteer.eval.capability import distinct_ngram_ratio

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUDGET_MMLU_DROP = 0.02
BUDGET_DISTINCT3 = 0.35
BUDGET_GSM8K_DROP = 0.05   # primary bound


def _ext_metric(results: dict, scale: float, metric: str):
    """A capability_ext panel value at a scale. Panel keys were floats,
    json-dumped to strings ("0.0", "1.0", "0.25"); match numerically so a
    "1" vs "1.0" formatting drift can never silently drop a measurement."""
    ext = results.get("capability_ext") or {}
    for k, v in ext.items():
        try:
            if abs(float(k) - float(scale)) < 1e-9:
                return (v or {}).get(metric)
        except (TypeError, ValueError):
            continue
    return None


def budget_verdict(rep, mmlu, base_mmlu, gsm8k, base_gsm8k, *,
                   legacy: bool = False,
                   gsm8k_drop: float = BUDGET_GSM8K_DROP):
    """The in-budget rule for one candidate scale -> True/False/None.

    Semantics: a DEFINITIVE failure on any measured metric is False;
    otherwise any required-but-unmeasured metric makes the candidate None
    ("not eligible", never in-budget); True only when every required check
    actually passed. legacy=True drops the GSM8K requirement (the
    stage-1-comparable map)."""
    if rep < BUDGET_DISTINCT3:
        return False
    if mmlu is not None and base_mmlu is not None \
            and mmlu < base_mmlu - BUDGET_MMLU_DROP:
        return False
    if not legacy and gsm8k is not None and base_gsm8k is not None \
            and gsm8k < base_gsm8k - gsm8k_drop:
        return False
    if mmlu is None or base_mmlu is None:
        return None
    if not legacy and (gsm8k is None or base_gsm8k is None):
        return None
    return True


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(payload, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# The artifact names stage2_run.py saves per arm (load_object dispatches
# on content, so trying both is enough).
ARTIFACT_NAMES = ("vector.safetensors", "object.pt")


def _cell_dirs(out: Path, ladder_dir: Path | None) -> list[Path]:
    cells = sorted(p.parent for p in out.glob("*/*/results.json"))
    if ladder_dir is not None:
        cells += sorted(p.parent for p in ladder_dir.glob("*/*/results.json"))
    return cells


def _load_cell_object(cell: Path, model, tok):
    from sisteer.methods import load_object

    for name in ARTIFACT_NAMES:
        path = cell / name
        if path.exists():
            return load_object(path, model, tok)
    raise FileNotFoundError(
        f"no intervention artifact under {cell} (tried {ARTIFACT_NAMES}); "
        "cannot measure candidate-scale MMLU for this cell"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="stage-2 matrix root: <out>/<instr>/<arm>/results.json")
    ap.add_argument("--ladder-dir", default=None,
                    help="optional ladder root (<...>/ladder) to fold rung "
                         "rows into the same frontier")
    ap.add_argument("--config", default=None,
                    help="experiment config yaml: model + model_revision + "
                         "dtype come from it via stage2_run's loader (D22#10)")
    ap.add_argument("--model", default=None,
                    help="override the config's model id")
    ap.add_argument("--revision", default=None,
                    help="override the config's pinned model_revision")
    ap.add_argument("--dtype", default=None,
                    help="override the config's dtype")
    ap.add_argument("--dry", action="store_true",
                    help="parse + repetition only; no model, no new "
                         "MMLU/GSM8K; writes frontier.preview.json, NEVER "
                         "frontier.json")
    ap.add_argument("--gsm8k-budget", type=float, default=BUDGET_GSM8K_DROP,
                    help="max GSM8K drop vs unsteered (D33; default 0.05)")
    ap.add_argument("--legacy-budget", action="store_true",
                    help="MMLU-only stage-1-comparable rule; writes "
                         "frontier.legacy.json, NEVER frontier.json")
    args = ap.parse_args()
    out = Path(args.out_dir)
    ladder_dir = Path(args.ladder_dir) if args.ladder_dir else None

    model = tok = mmlu_rows = gsm8k_rows = None
    gsm8k_cache_path = out / "frontier_gsm8k_fillins.json"
    gsm8k_cache: dict = (json.loads(gsm8k_cache_path.read_text())
                         if gsm8k_cache_path.exists() else {})
    if not args.dry:
        import yaml

        from sisteer.eval.capability_ext import _gsm8k_rows, _mmlu_rows
        from stage2_run import _load_model

        cfg = {}
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        if not (args.model or cfg.get("model")):
            raise SystemExit(
                "pass --config <experiment yaml> (or an explicit --model): "
                "the frontier MMLU must run the SAME pinned model snapshot "
                "and dtype as the matrix cells (D22#10) — no default model"
            )
        # The exact loader stage2_run uses, fed the config's model /
        # model_revision / dtype (+ attn_implementation), with CLI
        # overrides applied on top.
        eff = dict(cfg)
        if args.model:
            eff["model"] = args.model
        if args.revision:
            eff["model_revision"] = args.revision
        if args.dtype:
            eff["dtype"] = args.dtype
        model, tok = _load_model(eff)
        mmlu_rows = _mmlu_rows()
        if not args.legacy_budget:
            gsm8k_rows = _gsm8k_rows(n=200)

    cells = _cell_dirs(out, ladder_dir)
    if not cells:
        raise SystemExit(f"no */*/results.json under {out}"
                         + (f" or {ladder_dir}" if ladder_dir else ""))

    frontier: dict = {}
    methods_seen: list[str] = []
    for cell in cells:
        results = json.loads((cell / "results.json").read_text())
        instr = results.get("instruction", cell.parent.name)
        method = results.get("method", cell.name)
        if method not in methods_seen:
            methods_seen.append(method)
        arm_meta = results["arms"]
        records = results["records"]
        steered = [a for a in arm_meta if a["arm"] == "steered"]
        # None-safe: a cell without a prompted/baseline arm records None
        # rather than raising StopIteration.
        prompted = next((a for a in arm_meta if a["arm"] == "prompted"),
                        None)
        baseline = next((a for a in arm_meta if a["arm"] == "baseline"),
                        None)
        if not steered:
            continue
        best_c = max(a["compliance_strict"] for a in steered)
        cand = [a for a in steered
                if a["compliance_strict"] >= 0.9 * best_c
                and a["compliance_strict"] > 0]
        cap = results.get("capability", {})
        base_mmlu = cap.get("unsteered", {}).get("mmlu")
        # Unsteered GSM8K is object-free, hence model-global: prefer the
        # cell's own panel value, else the shared cached measurement.
        base_gsm8k = _ext_metric(results, 0.0, "gsm8k")
        if base_gsm8k is None and not args.legacy_budget:
            base_gsm8k = gsm8k_cache.get("unsteered")
            if base_gsm8k is None and not args.dry:
                from sisteer.eval.capability_ext import gsm8k_accuracy

                base_gsm8k = gsm8k_accuracy(
                    model, tok, None, scale=0.0, rows=gsm8k_rows)["acc"]
                gsm8k_cache["unsteered"] = base_gsm8k
                _write_json_atomic(gsm8k_cache_path, gsm8k_cache)

        obj = None
        per_scale = []
        for a in cand:
            scale = a["scale"]
            idx = arm_meta.index(a)
            responses = [r["response"] for r in records[idx]
                         if not r.get("nontrigger")]
            rep = distinct_ngram_ratio(responses)
            mmlu = cap.get(f"scale_{scale}", {}).get("mmlu")
            if mmlu is None and not args.dry:
                from sisteer.eval.capability_ext import mmlu_under_obj

                if obj is None:
                    obj = _load_cell_object(cell, model, tok)
                mmlu = mmlu_under_obj(model, tok, obj, scale=scale,
                                      rows=mmlu_rows)
            gsm8k = None
            if not args.legacy_budget:
                gsm8k = _ext_metric(results, scale, "gsm8k")
                cache_key = f"{instr}|{method}|{scale:g}"
                if gsm8k is None:
                    gsm8k = gsm8k_cache.get(cache_key)
                # A candidate already definitively failed (repetition or
                # MMLU) is False regardless of GSM8K — never pay the
                # generative fill-in (~200 x 512-token completions) for it.
                already_failed = (
                    rep < BUDGET_DISTINCT3
                    or (mmlu is not None and base_mmlu is not None
                        and mmlu < base_mmlu - BUDGET_MMLU_DROP))
                if gsm8k is None and not args.dry and not already_failed:
                    from sisteer.eval.capability_ext import gsm8k_accuracy

                    if obj is None:
                        obj = _load_cell_object(cell, model, tok)
                    gsm8k = gsm8k_accuracy(
                        model, tok, obj, scale=scale, rows=gsm8k_rows)["acc"]
                    # write-through: generative fill-ins are the expensive
                    # step, so a walltime kill must not re-pay them
                    gsm8k_cache[cache_key] = gsm8k
                    _write_json_atomic(gsm8k_cache_path, gsm8k_cache)
            # budget_verdict semantics: definitive failures are False,
            # required-but-unmeasured metrics make the candidate None
            # ("not eligible"), True only on full passes.
            in_budget = budget_verdict(
                rep, mmlu, base_mmlu, gsm8k, base_gsm8k,
                legacy=args.legacy_budget, gsm8k_drop=args.gsm8k_budget)
            per_scale.append({
                "scale": scale, "compliance": a["compliance_strict"],
                "compliance_loose": a["compliance_loose"],
                "mmlu": mmlu, "gsm8k": gsm8k, "distinct3": rep,
                "in_budget": in_budget,
            })
        ok = [p for p in per_scale if p["in_budget"] is True]
        best_ok = max(ok, key=lambda p: p["compliance"]) if ok else None
        frontier.setdefault(instr, {})[method] = {
            # None-safe: a cell without a prompted/baseline arm.
            "prompted": prompted["compliance_strict"] if prompted else None,
            "baseline": baseline["compliance_strict"] if baseline else None,
            "candidates": per_scale,
            "frontier": best_ok,
        }

    # --dry NEVER overwrites frontier.json: downstream consumers
    # (judge_batch pick_arms, the qwen-spot arm fill) read frontier.json,
    # and a parse-only rerun must not clobber a real MMLU-backed frontier.
    # --legacy-budget likewise writes its own file: the stage-1-comparable
    # map must never masquerade as the canonical frontier.
    target = out / "frontier.json"
    write_path = out / ("frontier.preview.json" if args.dry
                        else "frontier.legacy.json" if args.legacy_budget
                        else "frontier.json")
    write_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(write_path, frontier)
    if args.dry:
        print(f"DRY RUN: wrote {write_path} only; {target} untouched")

    print(f"\n{'instruction':22s} prompted  "
          + "  ".join(f"{m:>14s}" for m in methods_seen))
    for instr, cells_ in frontier.items():
        # a cell may carry prompted=None: print the first cell of the row
        # that has one.
        p = next((c["prompted"] for c in cells_.values()
                  if c["prompted"] is not None), None)
        ptxt = f"{p:.2f}" if p is not None else "  --"
        row = []
        for m in methods_seen:
            c = cells_.get(m)
            f = c and c["frontier"]
            row.append(f"{f['compliance']:.2f}@{f['scale']:g}" if f else "--")
        print(f"{instr:22s}   {ptxt}    "
              + "  ".join(f"{v:>14s}" for v in row))
    print(f"\nwrote {write_path}")


if __name__ == "__main__":
    main()
