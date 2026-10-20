"""Object-axis ablation: the same forward-KL objective compiled into
different intervention objects, everything else pinned to stage 1.

Cells (Llama-3.1-8B, stage-1 shared pairs, same eval battery):
  vector  — stage-1 distill_soft (already measured; read from stage-1 out)
  token   — one trainable soft input-embedding token (Steering Tokens /
            Wingate object), d params
  lora    — rank-1 LoRA on layer-15 o_proj (Prompt Baking's direction at
            matched-order size), 2d params

For each new object: train with train_object_kl on the SAME
pairs_to_distill_examples inputs stage-1 used, then evaluate four arms
(baseline / prompted / object@scale-sweep / both), with strict+loose
compliance, distinct-3gram, and MMLU under the intervention at every
steered scale. Output mirrors the stage-1 results.json layout.

Usage (array mode, one instruction per GPU job):
  uv run python scripts/object_ablation.py --instruction french
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sisteer.data import instructions as instr_registry
from sisteer.data.pairs import load_pairs, pairs_to_distill_examples
from sisteer.data.prompts import load_hf_prompts, split_prompts
from sisteer.eval import checkers
from sisteer.eval.capability import _mmlu_prompt, distinct_ngram_ratio
from sisteer.extraction.objects import (
    RankOneLoRAObject,
    SoftTokenObject,
    train_object_kl,
)
from sisteer.generate import generate_batch
from sisteer.models import build_chat_prompt, load_model_and_tokenizer

STAGE1_OUT = "$SISTEER_PROJECT_ROOT/out/stage1_llama8b"   # under the job environment's project directory, expanded in main()
LAYER = 15
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
INSTRUCTIONS = [
    "french", "lowercase", "concise", "json", "bullets",
    "include_notably", "no_sorry", "pirate", "multi_french_concise",
]  # non-conditional rows; the object comparison doesn't need the 2x2


def project_path(p: str) -> str:
    """``p`` with ``$SISTEER_PROJECT_ROOT`` (the cluster project directory the job environment names) expanded; refuses by name
    when ``p`` needs it and it is unset."""
    if "$SISTEER_PROJECT_ROOT" not in p:
        return p
    root = os.environ.get("SISTEER_PROJECT_ROOT")
    if not root:
        raise SystemExit("set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)")
    return p.replace("$SISTEER_PROJECT_ROOT", root)


def score(instruction, prompts, responses):
    records = []
    n_s = n_l = 0
    for p, r in zip(prompts, responses):
        s = checkers.check(instruction.checker, r, **instruction.checker_kwargs)
        l = checkers.check_loose(instruction.checker, r,
                                 **instruction.checker_kwargs)
        n_s += s
        n_l += l
        records.append({"prompt": p, "response": r, "strict": s, "loose": l})
    n = max(len(prompts), 1)
    return n_s / n, n_l / n, records


def mmlu_under(model, tok, obj=None, scale: float = 1.0,
               rows: list[dict] | None = None) -> float:
    """MMLU letter-logprob with an object intervention active (or none).
    Mirrors capability.mmlu_accuracy; the soft token additionally needs
    its sentinel injected into the prompt string."""
    import torch

    device = next(model.parameters()).device
    letter_ids = [tok(f" {c}", add_special_tokens=False)["input_ids"][-1]
                  for c in ["A", "B", "C", "D"]]
    ctx = obj.apply(model, scale=scale) if obj is not None else None
    correct = 0
    with torch.no_grad():
        if ctx is not None:
            ctx.__enter__()
        try:
            for row in rows:
                prompt = _mmlu_prompt(tok, row["question"], row["choices"])
                if obj is not None and obj.kind == "token":
                    prompt = obj.inject(prompt)
                enc = tok(prompt, return_tensors="pt").to(device)
                logits = model(**enc).logits[0, -1]
                pick = int(torch.tensor(
                    [logits[i] for i in letter_ids]).argmax())
                correct += pick == row["answer"]
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
    return correct / max(len(rows), 1)


def eval_object_arm(model, tok, instruction, prompts, *, arm, obj=None,
                    scale=1.0, batch_size=8, max_new_tokens=200):
    """eval_arm's logic with the object context replacing the sv hook."""
    system = instruction.text if arm in ("prompted", "both") else None
    rendered = [build_chat_prompt(tok, p, system=system) for p in prompts]
    use_obj = obj if arm in ("steered", "both") else None
    if use_obj is not None and use_obj.kind == "token":
        rendered = [use_obj.inject(r) for r in rendered]
    if use_obj is not None:
        with use_obj.apply(model, scale=scale):
            responses = generate_batch(
                model, tok, rendered, sv=None, batch_size=batch_size,
                max_new_tokens=max_new_tokens)
    else:
        responses = generate_batch(
            model, tok, rendered, sv=None, batch_size=batch_size,
            max_new_tokens=max_new_tokens)
    strict, loose, records = score(instruction, prompts, responses)
    return {
        "arm": arm,
        "scale": scale if use_obj is not None else 0.0,
        "compliance_strict": strict,
        "compliance_loose": loose,
        "distinct3": distinct_ngram_ratio(responses),
    }, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True, choices=INSTRUCTIONS)
    ap.add_argument("--objects", nargs="*", default=["token", "lora"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--stage1-out", default=str(STAGE1_OUT))
    ap.add_argument("--out-dir",
                    default="$SISTEER_PROJECT_ROOT/out/object_ablation")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    args.stage1_out, args.out_dir = project_path(args.stage1_out), project_path(args.out_dir)

    instruction = instr_registry.get(args.instruction)
    model, tok = load_model_and_tokenizer(args.model)

    # Same eval prompts as stage 1 (alpaca 600, hash split, first 100).
    base = load_hf_prompts("tatsu-lab/alpaca", n=600)
    _, eval_prompts = split_prompts(base, holdout_frac=0.3)
    eval_prompts = eval_prompts[: args.n_eval]

    pairs = load_pairs(Path(args.stage1_out) / args.instruction / "pairs.jsonl")
    examples = pairs_to_distill_examples(tok, pairs, instruction)

    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    ds = ds.shuffle(seed=0).select(range(200))
    mmlu_rows = [{"question": r["question"], "choices": r["choices"],
                  "answer": r["answer"]} for r in ds]

    out_root = Path(args.out_dir) / args.instruction
    for kind in args.objects:
        cell_dir = out_root / kind
        done = cell_dir / "results.json"
        if done.exists():
            print(f"skip {args.instruction}/{kind}: results.json exists")
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)

        if kind == "token":
            obj = SoftTokenObject(model, tok)
        elif kind == "lora":
            obj = RankOneLoRAObject(model, LAYER)
        else:
            raise ValueError(kind)

        history = train_object_kl(
            model, tok, obj, examples,
            epochs=args.epochs, batch_size=args.batch_size,
            log_fn=None,
        )
        obj.save(cell_dir / "object.pt")

        arms, all_records = [], []
        for arm in ("baseline", "prompted"):
            meta, recs = eval_object_arm(
                model, tok, instruction, eval_prompts, arm=arm)
            arms.append(meta)
            all_records.append(recs)
        capability = {
            "unsteered": {"mmlu": mmlu_under(model, tok, rows=mmlu_rows)}
        }
        for s in SCALES:
            meta, recs = eval_object_arm(
                model, tok, instruction, eval_prompts, arm="steered",
                obj=obj, scale=s)
            arms.append(meta)
            all_records.append(recs)
            capability[f"scale_{s}"] = {
                "mmlu": mmlu_under(model, tok, obj=obj, scale=s,
                                   rows=mmlu_rows)
            }
        both_scale = SCALES[len(SCALES) // 2]
        meta, recs = eval_object_arm(
            model, tok, instruction, eval_prompts, arm="both",
            obj=obj, scale=both_scale)
        arms.append(meta)
        all_records.append(recs)

        results = {
            "instruction": args.instruction,
            "category": instruction.category,
            "method": f"distill_soft_{kind}",
            "object": kind,
            "n_params": obj.n_params(),
            "layer": LAYER if kind == "lora" else None,
            "final_loss": history[-1] if history else None,
            "arms": arms,
            "capability": capability,
            "records": all_records,
        }
        done.write_text(json.dumps(results, indent=1))
        best = max((a for a in arms if a["arm"] == "steered"),
                   key=lambda a: a["compliance_strict"])
        print(f"{args.instruction}/{kind}: prompted="
              f"{arms[1]['compliance_strict']:.2f} best_steered="
              f"{best['compliance_strict']:.2f}@{best['scale']:g} "
              f"(distinct3={best['distinct3']:.2f}) params={obj.n_params()}")

    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
