"""Stage-1 post-processing: the within-budget compliance frontier.

The grid measures capability only at the best-compliance scale, so a cell
whose raw best sits at a degenerate scale can look DEGEN even though a
slightly lower scale was nearly as compliant and in-budget. This script
completes the picture per (instruction, method):

1. candidate scales = all scales within 0.9x of the cell's best strict
   compliance (and > 0);
2. MMLU at each candidate not already measured, reusing the stored vector;
3. repetition (distinct-3gram) of each candidate arm's recorded responses
   (free — from results.json);
4. budget: MMLU >= unsteered - 0.02 AND distinct3 >= 0.35;
5. frontier = best compliance among in-budget scales.

Prints the instruction x method frontier matrix (vs the prompted arm) and
writes frontier.json. --dry parses and scores repetition only (no GPU).

Usage: uv run python scripts/stage1_frontier.py --out-dir <stage1 out> [--dry]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sisteer.eval.capability import distinct_ngram_ratio

BUDGET_MMLU_DROP = 0.02
BUDGET_DISTINCT3 = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out/stage1_llama8b")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default=None,
                    help="pin a cached snapshot (gemma-4-31B-it offline load)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)

    model = tok = mmlu_rows = None
    if not args.dry:
        from datasets import load_dataset

        from sisteer.eval.capability import mmlu_accuracy
        from sisteer.models import load_model_and_tokenizer
        from sisteer.vectors import SteeringVector

        model, tok = load_model_and_tokenizer(args.model, revision=args.revision)
        ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.shuffle(seed=0).select(range(200))
        mmlu_rows = [
            {"question": r["question"], "choices": r["choices"],
             "answer": r["answer"]} for r in ds
        ]

    frontier: dict = {}
    for sfile in sorted(out.glob("summaries_*.json")):
        for s in json.loads(sfile.read_text()):
            instr, method = s["instruction"], s["method"]
            cell_dir = out / instr / method
            results = json.loads((cell_dir / "results.json").read_text())
            arms = results["records"]
            arm_meta = s["arms"]
            steered = [a for a in arm_meta if a["arm"] == "steered"]
            prompted = next(a for a in arm_meta if a["arm"] == "prompted")
            best_c = max(a["compliance_strict"] for a in steered)
            cand = [a for a in steered
                    if a["compliance_strict"] >= 0.9 * best_c
                    and a["compliance_strict"] > 0]
            cap = s.get("capability", {})
            base_mmlu = cap.get("unsteered", {}).get("mmlu")

            per_scale = []
            for a in cand:
                scale = a["scale"]
                # responses for this arm, from the full records (records is
                # a list per arm in arm order)
                idx = arm_meta.index(a)
                responses = [r["response"] for r in arms[idx]
                             if not r.get("nontrigger")]
                rep = distinct_ngram_ratio(responses)
                key = f"scale_{scale}"
                mmlu = cap.get(key, {}).get("mmlu")
                if mmlu is None and not args.dry:
                    sv = SteeringVector.load(cell_dir / "vector.safetensors")
                    mmlu = mmlu_accuracy(model, tok, sv=sv, scale=scale,
                                         rows=mmlu_rows)
                in_budget = (
                    rep >= BUDGET_DISTINCT3
                    and (mmlu is None or base_mmlu is None
                         or mmlu >= base_mmlu - BUDGET_MMLU_DROP)
                )
                per_scale.append({
                    "scale": scale, "compliance": a["compliance_strict"],
                    "false_fire": a.get("false_fire_strict"),
                    "mmlu": mmlu, "distinct3": rep, "in_budget": in_budget,
                })
            ok = [p for p in per_scale if p["in_budget"]
                  and (args.dry or p["mmlu"] is not None)]
            best_ok = max(ok, key=lambda p: p["compliance"]) if ok else None
            frontier.setdefault(instr, {})[method] = {
                "prompted": prompted["compliance_strict"],
                "prompted_false_fire": prompted.get("false_fire_strict"),
                "baseline": next(a for a in arm_meta
                                 if a["arm"] == "baseline")["compliance_strict"],
                "candidates": per_scale,
                "frontier": best_ok,
            }

    (out / "frontier.json").write_text(json.dumps(frontier, indent=1))

    methods = ["instr_diff", "caa", "actadd", "bidpo", "distill_soft",
               "distill_hard"]
    print(f"\n{'instruction':22s} prompted  " +
          "  ".join(f"{m:>10s}" for m in methods))
    for instr, cells in frontier.items():
        prompted = next(iter(cells.values()))["prompted"]
        row = []
        for m in methods:
            c = cells.get(m)
            f = c and c["frontier"]
            row.append(f"{f['compliance']:.2f}@{f['scale']:g}" if f else "--")
        print(f"{instr:22s}   {prompted:.2f}    " +
              "  ".join(f"{v:>10s}" for v in row))
    print(f"\nwrote {out / 'frontier.json'}")


if __name__ == "__main__":
    main()
